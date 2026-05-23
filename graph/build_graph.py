"""
Builds a heterogeneous transaction graph from the processed features dataset.

Nodes:
- Transaction nodes: one per transaction, features = engineered columns
- Card nodes: one per unique card (from card_id_str)
- Device nodes: one per unique device (from device_str)
- Email nodes: one per unique email domain (from email_str)

Edges:
- transaction -> card: each transaction connected to its card
- transaction -> device: each transaction connected to its device
- transaction -> email: each transaction connected to its email domain

Steps:
1. Load processed features from S3
2. Build transaction node features and labels
3. Build card, device, email nodes
4. Build all edges
5. Save graph to S3

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
    exclude_cols = [
        "TransactionID", "isFraud", "card_id_str",
        "device_str", "email_str", "TransactionDT",
        "card1", "card2"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    print(f"  Transaction features shape: {x.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.3%}")
    return x, y, feature_cols


# 3. Build entity nodes and index maps
def build_entity_nodes(df):
    print("Building entity nodes:")
    unique_cards = df["card_id_str"].unique()
    card_to_idx = {card: idx for idx, card in enumerate(unique_cards)}
    print(f"  Unique cards: {len(unique_cards)}")
    unique_devices = df["device_str"].unique()
    device_to_idx = {device: idx for idx, device in enumerate(unique_devices)}
    print(f"  Unique devices: {len(unique_devices)}")
    unique_emails = df["email_str"].unique()
    email_to_idx = {email: idx for idx, email in enumerate(unique_emails)}
    print(f"  Unique email domains: {len(unique_emails)}")
    return card_to_idx, device_to_idx, email_to_idx


# 4. Build edges
def build_edges(df, card_to_idx, device_to_idx, email_to_idx):
    print("Building edges:")
    tx_indices = torch.arange(len(df), dtype=torch.long)
    # transaction -> card edges
    card_indices = torch.tensor(
        [card_to_idx[c] for c in df["card_id_str"]], dtype=torch.long)
    card_edge_index = torch.stack([tx_indices, card_indices], dim=0)
    print(f"  Transaction->Card edges: {card_edge_index.shape[1]}")
    # transaction -> device edges
    device_indices = torch.tensor(
        [device_to_idx[d] for d in df["device_str"]], dtype=torch.long)
    device_edge_index = torch.stack([tx_indices, device_indices], dim=0)
    print(f"  Transaction->Device edges: {device_edge_index.shape[1]}")
    # transaction -> email edges
    email_indices = torch.tensor(
        [email_to_idx[e] for e in df["email_str"]], dtype=torch.long)
    email_edge_index = torch.stack([tx_indices, email_indices], dim=0)
    print(f"  Transaction->Email edges: {email_edge_index.shape[1]}")
    return card_edge_index, device_edge_index, email_edge_index


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
    card_to_idx, device_to_idx, email_to_idx = build_entity_nodes(df)
    card_edge_index, device_edge_index, email_edge_index = build_edges(
        df, card_to_idx, device_to_idx, email_to_idx)

    data = HeteroData()
    data["transaction"].x = x
    data["transaction"].y = y
    data["card"].num_nodes = len(card_to_idx)
    data["device"].num_nodes = len(device_to_idx)
    data["email"].num_nodes = len(email_to_idx)
    data["transaction", "uses", "card"].edge_index = card_edge_index
    data["transaction", "uses", "device"].edge_index = device_edge_index
    data["transaction", "uses", "email"].edge_index = email_edge_index

    print(f"\nGraph summary:")
    print(f"  Transaction nodes: {data['transaction'].x.shape[0]}")
    print(f"  Card nodes: {data['card'].num_nodes}")
    print(f"  Device nodes: {data['device'].num_nodes}")
    print(f"  Email nodes: {data['email'].num_nodes}")
    print(f"  Card edges: {data['transaction', 'uses', 'card'].edge_index.shape[1]}")
    print(f"  Device edges: {data['transaction', 'uses', 'device'].edge_index.shape[1]}")
    print(f"  Email edges: {data['transaction', 'uses', 'email'].edge_index.shape[1]}")

    save_graph_to_s3(data)
    print("\nDone!")