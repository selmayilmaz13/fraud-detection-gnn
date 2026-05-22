"""
Builds a heterogeneous transaction graph from the processed features dataset.

Nodes:
- Transaction nodes: one per transaction, features = engineered columns
- Card nodes: one per unique card (from card_id_str)

Edges:
- transaction -> card: each transaction is connected to its card

Steps:
1. Load processed features from S3
2. Build transaction node features and labels
3. Build card nodes from unique card_id_str values
4. Build edges between transactions and cards
5. Save graph to S3 as a .pt file

Output: s3://fraud-detection-gnn/processed/graph.pt
"""

import boto3
import torch
import pandas as pd
import numpy as np
from io import StringIO
from torch_geometric.data import HeteroData

BUCKET_NAME = "fraud-detection-gnn"

# 1. Load features from S3
def load_features():
    print("Loading features from S3:")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# 2. Build transaction node features and labels
def build_transaction_nodes(df):
    print("Building transaction nodes:")
    # columns to exclude from node features
    exclude_cols = [
        "TransactionID", "isFraud", "card_id_str",
        "TransactionDT", "card1", "card2"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    print(f"  Transaction features shape: {x.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.3%}")
    return x, y, feature_cols


# 3. Build card nodes
def build_card_nodes(df):
    print("Building card nodes:")
    # get unique cards and map them to indices
    unique_cards = df["card_id_str"].unique()
    card_to_idx = {card: idx for idx, card in enumerate(unique_cards)}
    print(f"  Unique cards: {len(unique_cards)}")
    return card_to_idx


# 4. Build edges between transactions and cards
def build_edges(df, card_to_idx):
    print("Building edges:")
    # each transaction connects to its card
    transaction_indices = torch.arange(len(df), dtype=torch.long)
    card_indices = torch.tensor(
        [card_to_idx[card] for card in df["card_id_str"]],dtype=torch.long)
    # edge index shape: [2, num_edges]
    edge_index = torch.stack([transaction_indices, card_indices], dim=0)

    print(f"  Edges: {edge_index.shape[1]}")
    return edge_index

# 5. Save graph to S3
def save_graph_to_s3(data, s3_key="processed/graph.pt"):
    print(f"Saving graph to s3://{BUCKET_NAME}/{s3_key}...")
    torch.save(data, "/tmp/graph.pt")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/graph.pt", BUCKET_NAME, s3_key)
    print(f"  Saved to s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    df = load_features()
    x, y, feature_cols = build_transaction_nodes(df)
    card_to_idx = build_card_nodes(df)
    edge_index = build_edges(df, card_to_idx)
    # build heterogeneous graph
    data = HeteroData()
    # transaction nodes
    data["transaction"].x = x
    data["transaction"].y = y
    # card nodes (no features, just identity)
    data["card"].num_nodes = len(card_to_idx)
    # edges: transaction -> card
    data["transaction", "uses", "card"].edge_index = edge_index
    print(f"\nGraph summary:")
    print(f"  Transaction nodes: {data['transaction'].x.shape[0]}")
    print(f"  Card nodes: {data['card'].num_nodes}")
    print(f"  Edges: {data['transaction', 'uses', 'card'].edge_index.shape[1]}")

    save_graph_to_s3(data)
    print("\nDone!")