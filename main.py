from src.inference import load_model, predict_fraud

# Load the trained model
model = load_model("path/to/trained_model.pkl")

input_data = {
    "TransactionDT": 172800,
    "TransactionAmt": 250.75,
    "ProductCD": "C",
    "card1": 1345,
    "card2": 210.0,
    "card3": 150.0,
    "card4": "mastercard",
    "card5": 166.0,
    "card6": "debit",
    "addr1": 120.0,
    "addr2": 87.0,
    "dist1": 10.0,
    "dist2": 3.0,
    "P_emaildomain": "yahoo.com",
    "R_emaildomain": "yahoo.com",
    "DeviceType": "mobile",
    "DeviceInfo": "iOS"
}

# Pass the model to predict_fraud
result = predict_fraud(input_data, model=model)
print(f"Fraud prediction result: {result}")