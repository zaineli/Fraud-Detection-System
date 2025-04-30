import os
import logging
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from datetime import datetime
from inference import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configurable paths
DATA_DIR = "./data"
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

class DetailedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_cols = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
        ]
        self.num_cols = [
            'TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5',
            'addr1', 'addr2', 'dist1', 'dist2'
        ]
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def fit(self, X, y=None):
        self.imputer.fit(X[self.num_cols])
        self.scaler.fit(self.imputer.transform(X[self.num_cols]))
        for col in self.cat_cols:
            le = LabelEncoder()
            X_col = X[col].astype(str).fillna('missing')
            le.fit(X_col)
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        X_num = self.imputer.transform(X[self.num_cols])
        X_num = self.scaler.transform(X_num)
        for idx, col in enumerate(self.num_cols):
            X[col] = X_num[:, idx]
        for col in self.cat_cols:
            le = self.label_encoders[col]
            X[col] = le.transform(X[col].astype(str).fillna('missing'))
        X = X.fillna(-999)
        return X

def load_data():
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"), nrows=1000000)
    idt = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"), nrows=1000000)
    df = txn.merge(idt, on="TransactionID", how="left")
    selected_columns = [
        'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD',
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2',
        'P_emaildomain', 'R_emaildomain',
        'DeviceType', 'DeviceInfo',
        'isFraud'
    ]
    df = df[selected_columns]
    logging.info(f"Loaded data with shape: {df.shape}")
    return df

def augment_data(X, y):
    sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.2)
    X_res, y_res = sm.fit_resample(X, y)
    logging.info(f"After SMOTE: X shape {X_res.shape}, y distribution: {np.bincount(y_res)}")
    return X_res, y_res

def cross_validate_model(pipeline, X, y, cv=3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    aucs = cross_val_score(
        pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1
    )
    logging.info(f"Cross-validated AUCs: {aucs}, Mean AUC: {aucs.mean():.4f}")
    return aucs

def create_models():
    """Create multiple models for fraud detection"""
    models = {
        "xgb": Pipeline([
            ("features", FeatureEngineer()),
            ("model", xgb.XGBClassifier(
                n_estimators=500,
                max_depth=12,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='auc',
                tree_method='hist',
                use_label_encoder=False,
                n_jobs=-1
            ))
        ]),
        
        "rf": Pipeline([
            ("features", FeatureEngineer()),
            ("model", RandomForestClassifier(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        
        "gbm": Pipeline([
            ("features", FeatureEngineer()),
            ("model", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=RANDOM_STATE
            ))
        ]),
        
        "lr": Pipeline([
            ("features", FeatureEngineer()),
            ("model", LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        
        "voting": Pipeline([
            ("features", FeatureEngineer()),
            ("model", VotingClassifier(
                estimators=[
                    ('xgb', xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=10,
                        learning_rate=0.03,
                        subsample=0.8,
                        random_state=RANDOM_STATE,
                        eval_metric='auc',
                        use_label_encoder=False
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=RANDOM_STATE
                    )),
                    ('gbm', GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=RANDOM_STATE
                    ))
                ],
                voting='soft',
                n_jobs=-1
            ))
        ])
    }
    
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with detailed metrics"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        "auc": auc,
        "avg_precision": avg_precision,
        "optimal_threshold": optimal_threshold,
        "best_f1": f1_scores[optimal_idx]
    }

def train():
    df = load_data()
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])

    # Preprocessing
    preprocessor = DetailedPreprocessor()
    X_prep = preprocessor.fit_transform(X)
    logging.info("Preprocessing complete.")

    # Data augmentation (SMOTE)
    X_aug, y_aug = augment_data(X_prep, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, stratify=y_aug, test_size=0.2, random_state=RANDOM_STATE
    )

    # Create and train multiple models
    models = create_models()
    best_score = 0
    best_model_name = None
    results = {}

    for name, pipeline in models.items():
        logging.info(f"Training {name} model...")
        
        # Cross-validation
        aucs = cross_validate_model(pipeline, X_train, y_train, cv=3)
        
        # Final training
        pipeline.fit(X_train, y_train)
        
        # Evaluation
        eval_results = evaluate_model(pipeline, X_test, y_test)
        results[name] = eval_results
        
        logging.info(f"{name} model - Test AUC: {eval_results['auc']:.4f}, Avg Precision: {eval_results['avg_precision']:.4f}")
        
        if eval_results['auc'] > best_score:
            best_score = eval_results['auc']
            best_model_name = name

    # Save all models with performance metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, pipeline in models.items():
        model_path = os.path.join(MODEL_DIR, f"{name}_fraud_model_{timestamp}.joblib")
        joblib.dump({
            "pipeline": pipeline, 
            "preprocessor": preprocessor,
            "metrics": results[name]
        }, model_path)
        logging.info(f"{name} model saved to {model_path}")

    logging.info(f"Best performing model: {best_model_name} with AUC: {best_score:.4f}")

if __name__ == "__main__":
    train()