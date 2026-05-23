"""
Generates feature importance plots for the XGBoost fraud detection model.
Uses XGBoost's built-in feature importance and a simple prediction-based
approach to explain which features drive fraud predictions.

Plots generated:
1. Bar plot — top 20 features by importance
2. Gain plot — top 20 features by information gain
3. Sample predictions — fraud vs non-fraud feature comparison

Steps:
1. Load processed features from S3
2. Load XGBoost model from S3
3. Generate importance plots
4. Save plots to S3

Output: s3://fraud-detection-gnn/outputs/*.png
"""

import boto3
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

BUCKET_NAME = "fraud-detection-gnn"
TRAIN_RATIO = 0.8


# 1. Load features from S3
def load_features():
    print("Loading features from S3:")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. Load model from S3
def load_model():
    print("Loading XGBoost model from S3:")
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, "models/xgboost_baseline.pkl", "/tmp/xgboost_baseline.pkl")
    model = joblib.load("/tmp/xgboost_baseline.pkl")
    print("  Model loaded successfully")
    return model


# 3. Prepare features
def prepare_features(df):
    print("Preparing features:")
    exclude_cols = [
        "TransactionID", "isFraud", "card_id_str",
        "device_str", "email_str", "TransactionDT",
        "card1", "card2"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    split_idx = int(len(df) * TRAIN_RATIO)
    X_test = df[feature_cols].iloc[split_idx:]
    y_test = df["isFraud"].iloc[split_idx:]
    print(f"  Test set: {X_test.shape[0]} rows, {y_test.sum()} fraud")
    return X_test, y_test, feature_cols


# 4. Save plot to S3
def save_plot_to_s3(filename):
    s3 = boto3.client("s3")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    s3_key = f"outputs/{filename}"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=buf.getvalue())
    print(f"  Saved to s3://{BUCKET_NAME}/{s3_key}")
    buf.close()
    plt.close()


# 5. Plot feature importance by weight
def plot_importance_weight(model, feature_cols):
    print("Generating feature importance (weight) plot:")
    importance = model.get_booster().get_score(importance_type="weight")
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title("Top 20 Features by Weight — XGBoost Fraud Detection", fontsize=14)
    ax.set_xlabel("Feature Weight (number of times used in trees)")
    plt.tight_layout()
    save_plot_to_s3("feature_importance_weight.png")


# 6. Plot feature importance by gain
def plot_importance_gain(model, feature_cols):
    print("Generating feature importance (gain) plot:")
    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title("Top 20 Features by Gain — XGBoost Fraud Detection", fontsize=14)
    ax.set_xlabel("Feature Gain (average improvement in accuracy)")
    plt.tight_layout()
    save_plot_to_s3("feature_importance_gain.png")


# 7. Plot fraud vs non-fraud feature distributions
def plot_feature_distributions(model, X_test, y_test, feature_cols):
    print("Generating fraud vs non-fraud feature distributions:")

    # get top 6 features by gain
    importance = model.get_booster().get_score(importance_type="gain")
    top_features = sorted(importance, key=importance.get, reverse=True)[:6]
    top_features = [f for f in top_features if f in feature_cols][:6]
    fraud = X_test[y_test == 1]
    legit = X_test[y_test == 0].sample(n=len(fraud), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        axes[i].hist(legit[feat].clip(-10, 10), bins=50, alpha=0.6, label="Legit", color="blue")
        axes[i].hist(fraud[feat].clip(-10, 10), bins=50, alpha=0.6, label="Fraud", color="red")
        axes[i].set_title(feat, fontsize=12)
        axes[i].legend()

    plt.suptitle("Feature Distributions: Fraud vs Legitimate Transactions", fontsize=14)
    plt.tight_layout()
    save_plot_to_s3("feature_distributions.png")


if __name__ == "__main__":
    df = load_features()
    model = load_model()
    X_test, y_test, feature_cols = prepare_features(df)
    plot_importance_weight(model, feature_cols)
    plot_importance_gain(model, feature_cols)
    plot_feature_distributions(model, X_test, y_test, feature_cols)
    print("\nDone! All plots saved to S3.")