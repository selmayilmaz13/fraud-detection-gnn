"""
Trains an XGBoost baseline model on flat transaction features
(no graph structure) to benchmark against the GNN.

Steps:
1. Load processed features from S3
2. Time-based train/test split
3. Handle class imbalance with scale_pos_weight
4. Train XGBoost
5. Evaluate with AUC-ROC, Recall, Precision
6. Save model to S3

Output: s3://fraud-detection-gnn/models/xgboost_baseline.pkl
"""

import boto3
import joblib
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score,precision_score, classification_report)
from xgboost import XGBClassifier

BUCKET_NAME = "fraud-detection-gnn"

# 1. Load features from S3
def load_features():
    print("Loading features from S3:")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# 2. Train/test split
def split_data(df):
    print("Splitting data:")
    exclude_cols = ["TransactionID", "isFraud", "card_id_str","TransactionDT", "card1", "card2"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df["isFraud"]

    # time-based split — first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {X_train.shape[0]} rows, {y_train.sum()} fraud")
    print(f"  Test:  {X_test.shape[0]} rows, {y_test.sum()} fraud")
    return X_train, X_test, y_train, y_test

# 3. Train XGBoost
def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost baseline:")
    # scale_pos_weight handles class imbalance
    # ratio of negative to positive samples
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50)
    return model

# 4. Evaluate model
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(classification_report(y_test, y_pred))
    return auc, recall, precision

# 5. Save model to S3
def save_model_to_s3(model):
    print("Saving model to S3:")
    joblib.dump(model, "/tmp/xgboost_baseline.pkl")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/xgboost_baseline.pkl", BUCKET_NAME, "models/xgboost_baseline.pkl")
    print(f"  Saved to s3://{BUCKET_NAME}/models/xgboost_baseline.pkl")


if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_xgboost(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    save_model_to_s3(model)
    print("\nDone!")