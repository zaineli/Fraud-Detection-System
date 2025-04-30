# src/inference.py

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_PATH = "model/ensemble_fraud_pipeline.joblib"

CATEGORICAL_COLS = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo'
]

NUMERIC_COLS = [
    'TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3',
    'card5', 'addr1', 'addr2', 'dist1', 'dist2'
]

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save mode values of categorical for transform
        self.fill_values_ = {}
        for col in CATEGORICAL_COLS:
            self.fill_values_[col] = X[col].mode(dropna=True)[0] if col in X else 'unknown'
        
        # For numeric columns, use median for imputation
        for col in NUMERIC_COLS:
            self.fill_values_[col] = X[col].median() if col in X else 0
            
        # Store column statistics for feature scaling
        self.num_stats_ = {}
        for col in NUMERIC_COLS:
            if col in X:
                self.num_stats_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std() if X[col].std() > 0 else 1.0,
                    'min': X[col].min(),
                    'max': X[col].max()
                }
                
        return self

    def transform(self, X):
        X = X.copy()

        # Handle categorical features
        for col in CATEGORICAL_COLS:
            if col in X:
                X[col] = X[col].fillna(self.fill_values_.get(col, 'unknown')).astype(str)
                # Reduce cardinality for rare categories
                value_counts = X[col].value_counts()
                rare_categories = value_counts[value_counts < 10].index
                X.loc[X[col].isin(rare_categories), col] = 'RARE'

        # Handle numeric features
        for col in NUMERIC_COLS:
            if col in X:
                X[col] = X[col].fillna(self.fill_values_.get(col, 0))
                
                # Clip outliers - values beyond 3 std devs
                if col in self.num_stats_:
                    mean, std = self.num_stats_[col]['mean'], self.num_stats_[col]['std']
                    X[col] = X[col].clip(mean - 3*std, mean + 3*std)
                
                # Normalize numeric features
                if col in self.num_stats_:
                    X[col] = (X[col] - self.num_stats_[col]['mean']) / self.num_stats_[col]['std']
        
        # Add interaction features
        if 'TransactionAmt' in X and 'card1' in X:
            X['TransactionAmt_to_card1_ratio'] = X['TransactionAmt'] / (X['card1'] + 1)
            
        # Time-based features if TransactionDT exists
        if 'TransactionDT' in X:
            # Convert to hours/days for better patterns
            X['Transaction_hour'] = (X['TransactionDT'] / 3600) % 24
            X['Transaction_day'] = (X['TransactionDT'] / (3600 * 24)) % 7

        # One-hot encoding for categoricals
        X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

        return X

def predict(input_data, threshold=0.5):
    """
    Predicts fraud probability and label from input data.
    Supports either:
    - A single dictionary
    - A pandas DataFrame
    - A path to a CSV file
    
    Args:
        input_data: The data to predict on
        threshold: Probability threshold for binary classification
    """
    print("Loading model...")
    pipeline = joblib.load(MODEL_PATH)

    # Load and prepare input
    if isinstance(input_data, str):  # CSV path
        data = pd.read_csv(input_data)
    elif isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data.copy()
    else:
        raise ValueError("Unsupported input type. Use dict, DataFrame, or CSV file path.")

    # Check for required columns
    missing = set(ALL_FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Make predictions
    proba = pipeline.predict_proba(data)[:, 1]
    # Use custom threshold instead of default 0.5
    pred = (proba >= threshold).astype(int)

    # Add results to output
    result = data.copy()
    result["isFraud_proba"] = proba
    result["isFraud_pred"] = pred
    
    # Add confidence level categorization
    result["confidence"] = pd.cut(
        result["isFraud_proba"], 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=["low_risk", "medium_risk", "high_risk"]
    )

    # Return important columns first
    cols_order = ["isFraud_proba", "isFraud_pred", "confidence"] + list(data.columns)
    return result[cols_order]
