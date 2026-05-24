"""
AWS Lambda handler for the fraud detection API.
Accepts a POST request with transaction features and returns
a fraud risk score with top contributing features.

Expected request body:
{
    "TransactionAmt": 150.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop",
    "card_fraud_rate": 0.02,
    "card_tx_count": 5
}

Response:
{
    "fraud_probability": 0.23,
    "risk_level": "Low",
    "top_features": [
        {"feature": "card_fraud_rate", "value": 0.02},
        {"feature": "TransactionAmt", "value": 150.0}
    ]
}
"""

import json
import boto3
import numpy as np
from xgboost import XGBClassifier

BUCKET_NAME = "fraud-detection-gnn"
MODEL_KEY = "models/xgboost_baseline.pkl"
TMP_MODEL_PATH = "/tmp/xgboost_baseline.pkl"

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"}

RISK_THRESHOLDS = {
    "Low": (0.0, 0.3),
    "Medium": (0.3, 0.6),
    "High": (0.6, 1.0)}

# cached model — only loads once per Lambda container
_model = None


# 1. Load model from S3
def load_model():
    global _model
    if _model is None:
        print("Loading model from S3...")
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, "models/xgboost_baseline.json", "/tmp/xgboost_baseline.json")
        _model = XGBClassifier()
        _model.load_model("/tmp/xgboost_baseline.json")
        print("Model loaded successfully")
    return _model


# 2. Get risk level from probability
def get_risk_level(prob):
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= prob <= high:
            return level
    return "High"


# 3. Get top contributing features
def get_top_features(model, X, feature_names, n=5):
    importance = model.get_booster().get_score(importance_type="gain")
    top = sorted(importance, key=importance.get, reverse=True)[:n]
    result = []
    for feat in top:
        if feat in feature_names:
            idx = feature_names.index(feat)
            result.append({
                "feature": feat,
                "value": round(float(X[0][idx]), 4)})
    return result


# 4. Preprocess input
def preprocess_input(body, model):
    feature_names = model.get_booster().feature_names
    row = {col: 0.0 for col in feature_names}
    cat_mappings = {
        "ProductCD": {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4},
        "card4": {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3},
        "card6": {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3},}

    for key, val in body.items():
        if key in row and key not in cat_mappings:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass

    for col, mapping in cat_mappings.items():
        if col in body and col in row:
            row[col] = float(mapping.get(body[col], 0))

    import numpy as np
    X = np.array([[row[col] for col in feature_names]], dtype=np.float32)
    return X, feature_names


# 5. Main Lambda handler
def handler(event, context):
    try:
        # handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": HEADERS,
                "body": ""}

        body = json.loads(event.get("body", "{}"))

        if not body:
            return {
                "statusCode": 400,
                "headers": HEADERS,
                "body": json.dumps({"error": "request body is required"})}

        model = load_model()
        X, feature_cols = preprocess_input(body, model)

        fraud_prob = float(model.predict_proba(X)[0][1])
        risk_level = get_risk_level(fraud_prob)
        top_features = get_top_features(model, X, feature_cols)

        response = {
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "top_features": top_features}

        return {
            "statusCode": 200,
            "headers": HEADERS,
            "body": json.dumps(response)}

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": HEADERS,
            "body": json.dumps({"error": str(e)})}