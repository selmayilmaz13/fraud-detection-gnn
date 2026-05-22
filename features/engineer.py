"""
Loads, merges, and engineers features from the IEEE-CIS fraud detection dataset.

Steps:
1. Load transaction and identity data from data/raw
2. Merge on TransactionID (left join: identity only covers 24% of transactions)
3. Drop features with >80% missing values
4. Handle remaining missing values
5. Engineer new features:
   - log(TransactionAmt) to handle right skew
   - hour and day from TransactionDT
   - card_id as unique card identifier (card1 + card2)
6. Encode categorical features
7. Save processed features to data/processed/features.csv

Key findings from EDA:
- Fraud rate: 3.5%, so severe class imbalance
- ProductCD is the strongest predictor (C has 11.7% fraud rate)
- Credit cards have 3x higher fraud rate than debit
- 47 V features have >80% missing — drop them
- 180 V features have <50% missing — keep them
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
MISSING_THRESHOLD = 0.8


# 1. Load data
def load_data():
    print("Loading data:")
    transactions = pd.read_csv(f"{RAW_PATH}/train_transaction.csv")
    identity = pd.read_csv(f"{RAW_PATH}/train_identity.csv")
    print(f"  Transactions: {transactions.shape}")
    print(f"  Identity: {identity.shape}")
    return transactions, identity

# 2. Merge datasets
def merge_data(transactions, identity):
    print("Merging datasets:")
    df = transactions.merge(identity, on="TransactionID", how="left")
    df = df.copy()
    print(f"  Merged shape: {df.shape}")
    return df

# 3. Drop high missing features
def drop_high_missing(df):
    print("Dropping high missing features:")
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} features with >{MISSING_THRESHOLD*100}% missing")
    print(f"  Remaining features: {df.shape[1]}")
    return df

# 4. Handle remaining missing values
def handle_missing(df):
    print("Handling remaining missing values:")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[cat_cols] = df[cat_cols].fillna("unknown")
    print(f"  Remaining missing values: {df.isnull().sum().sum()}")
    return df

# 5. Engineer new features
def engineer_features(df):
    print("Engineering features:")
    # log transform
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    # extract hour and day from TransactionDT (seconds since reference)
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
    # unique card identifier (card1 and card2)
    df["card_id"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
    # save string version before encoding (for graph construction)
    df["card_id_str"] = df["card_id"]
    # transaction count per card 
    card_counts = df.groupby("card_id")["TransactionID"].transform("count")
    df["card_tx_count"] = card_counts
    # mean fraud rate per card (how often this card is associated with fraud)
    card_fraud_rate = df.groupby("card_id")["isFraud"].transform("mean")
    df["card_fraud_rate"] = card_fraud_rate

    print(f"  Features after engineering: {df.shape[1]}")
    return df

# 6. Encode categorical features
def encode_categoricals(df):
    print("Encoding categorical features:")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # keep card_id_str as is
    cat_cols = [c for c in cat_cols if c != "card_id_str"]
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]
    print(f"  Encoded {len(cat_cols)} categorical columns")
    return df

# 7. Save processed data
def save_data(df):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    output_path = f"{PROCESSED_PATH}/features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.3%}")


if __name__ == "__main__":
    transactions, identity = load_data()
    df = merge_data(transactions, identity)
    df = drop_high_missing(df)
    df = handle_missing(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_data(df)
    print("\nDone!")