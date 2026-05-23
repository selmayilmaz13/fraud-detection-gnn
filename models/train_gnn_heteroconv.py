"""
Trains a Heterogeneous GNN using PyG's HeteroConv on the transaction graph.
Each edge type (card, device, email) gets its own SAGEConv layer with
separate learned weights. This is the correct way to handle heterogeneous
graphs with multiple node and edge types.

Architecture:
- HeteroConv layer 1: transaction(361) + card/device/email(0) -> 256
- ReLU + Dropout
- HeteroConv layer 2: 256 -> 128
- ReLU
- Linear classifier on transaction nodes: 128 -> 2
- Weighted cross-entropy loss for class imbalance

Steps:
1. Load graph from S3
2. Time-based train/test mask
3. Define HeteroGNN model
4. Train with weighted loss and early stopping
5. Evaluate with AUC-ROC, Recall, Precision
6. Save model to S3

Output: s3://fraud-detection-gnn/models/gnn_heteroconv.pt
"""

import boto3
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report

BUCKET_NAME = "fraud-detection-gnn"
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT = 0.5
LEARNING_RATE = 0.0005
EPOCHS = 150
TRAIN_RATIO = 0.8
WEIGHT_DECAY = 1e-4

# 1. Load graph from S3
def load_graph():
    print("Loading graph from S3:")
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, "processed/graph.pt", "/tmp/graph.pt")
    data = torch.load("/tmp/graph.pt", weights_only=False)
    print(f"  Transaction nodes: {data['transaction'].x.shape[0]}")
    print(f"  Card nodes: {data['card'].num_nodes}")
    print(f"  Device nodes: {data['device'].num_nodes}")
    print(f"  Email nodes: {data['email'].num_nodes}")
    return data


# 2. Build train/test masks
def build_masks(num_nodes):
    print("Building train/test masks:")
    split_idx = int(num_nodes * TRAIN_RATIO)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:split_idx] = True
    test_mask[split_idx:] = True
    print(f"  Train: {train_mask.sum().item()} nodes")
    print(f"  Test:  {test_mask.sum().item()} nodes")
    return train_mask, test_mask


# 3. Define HeteroGNN model
class FraudHeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, num_cards, num_devices, num_emails):
        super(FraudHeteroGNN, self).__init__()
        # layer 1 (separate SAGEConv for each edge type)
        self.conv1 = HeteroConv({
            ("transaction", "uses", "card"): SAGEConv((-1, -1), hidden1),
            ("transaction", "uses", "device"): SAGEConv((-1, -1), hidden1),
            ("transaction", "uses", "email"): SAGEConv((-1, -1), hidden1),
            ("card", "used_by", "transaction"): SAGEConv((-1, -1), hidden1),
            ("device", "used_by", "transaction"): SAGEConv((-1, -1), hidden1),
            ("email", "used_by", "transaction"): SAGEConv((-1, -1), hidden1),
        }, aggr="sum")

        # layer 2
        self.conv2 = HeteroConv({
            ("transaction", "uses", "card"): SAGEConv((-1, -1), hidden2),
            ("transaction", "uses", "device"): SAGEConv((-1, -1), hidden2),
            ("transaction", "uses", "email"): SAGEConv((-1, -1), hidden2),
            ("card", "used_by", "transaction"): SAGEConv((-1, -1), hidden2),
            ("device", "used_by", "transaction"): SAGEConv((-1, -1), hidden2),
            ("email", "used_by", "transaction"): SAGEConv((-1, -1), hidden2),
        }, aggr="sum")

        self.classifier = torch.nn.Linear(hidden2, 2)
        self.dropout = torch.nn.Dropout(DROPOUT)

        # learnable embeddings for card, device, email nodes
        self.card_emb = torch.nn.Embedding(num_cards, hidden1)
        self.device_emb = torch.nn.Embedding(num_devices, hidden1)
        self.email_emb = torch.nn.Embedding(num_emails, hidden1)

    def forward(self, x_dict, edge_index_dict):
        # layer 1
        out = self.conv1(x_dict, edge_index_dict)
        out = {key: F.relu(val) for key, val in out.items()}
        out["transaction"] = self.dropout(out["transaction"])
        # layer 2
        out = self.conv2(out, edge_index_dict)
        out = {key: F.relu(val) for key, val in out.items()}
        out["transaction"] = self.dropout(out["transaction"])

        return self.classifier(out["transaction"])


# 4. Train model
def train_model(data, train_mask, test_mask):
    print("Training HeteroGNN:")

    x = data["transaction"].x
    y = data["transaction"].y

    # normalize transaction features
    scaler = StandardScaler()
    x_scaled = torch.tensor(
        scaler.fit_transform(x.numpy()),
        dtype=torch.float)

    num_cards = data["card"].num_nodes
    num_devices = data["device"].num_nodes
    num_emails = data["email"].num_nodes

    # build x_dict with transaction features and embeddings for other nodes
    card_ids = torch.arange(num_cards)
    device_ids = torch.arange(num_devices)
    email_ids = torch.arange(num_emails)
    edge_index_dict = {
        ("transaction", "uses", "card"): data["transaction", "uses", "card"].edge_index,
        ("transaction", "uses", "device"): data["transaction", "uses", "device"].edge_index,
        ("transaction", "uses", "email"): data["transaction", "uses", "email"].edge_index,
        # reverse edges so transactions receive messages back
        ("card", "used_by", "transaction"): data["transaction", "uses", "card"].edge_index.flip(0),
        ("device", "used_by", "transaction"): data["transaction", "uses", "device"].edge_index.flip(0),
        ("email", "used_by", "transaction"): data["transaction", "uses", "email"].edge_index.flip(0),}

    n_neg = (y == 0).sum().item()
    n_pos = (y == 1).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)
    print(f"  Class weight for fraud: {n_neg/n_pos:.1f}")

    model = FraudHeteroGNN(
        in_channels=x_scaled.shape[1],
        hidden1=HIDDEN_DIM_1,
        hidden2=HIDDEN_DIM_2,
        num_cards=num_cards,
        num_devices=num_devices,
        num_emails=num_emails)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_auc = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        # build x_dict fresh each forward pass
        x_dict = {
            "transaction": x_scaled,
            "card": model.card_emb(card_ids),
            "device": model.device_emb(device_ids),
            "email": model.email_emb(email_ids),}

        out = model(x_dict, edge_index_dict)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x_dict = {
                    "transaction": x_scaled,
                    "card": model.card_emb(card_ids),
                    "device": model.device_emb(device_ids),
                    "email": model.email_emb(email_ids),}
                out = model(x_dict, edge_index_dict)
                prob = F.softmax(out, dim=1)[:, 1]
                test_auc = roc_auc_score(
                    y[test_mask].numpy(),
                    prob[test_mask].numpy())
            if test_auc > best_auc:
                best_auc = test_auc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Test AUC: {test_auc:.4f}")

    print(f"  Best AUC: {best_auc:.4f}")
    model.load_state_dict(best_model_state)
    return model, x_scaled, y, edge_index_dict, card_ids, device_ids, email_ids


# 5. Evaluate model
def evaluate_model(model, x_scaled, y, edge_index_dict, card_ids, device_ids, email_ids, test_mask):
    print("\nEvaluating HeteroGNN:")
    model.eval()
    with torch.no_grad():
        x_dict = {
            "transaction": x_scaled,
            "card": model.card_emb(card_ids),
            "device": model.device_emb(device_ids),
            "email": model.email_emb(email_ids),}
        out = model(x_dict, edge_index_dict)
        pred = out.argmax(dim=1)
        prob = F.softmax(out, dim=1)[:, 1]

    y_test = y[test_mask].numpy()
    y_pred = pred[test_mask].numpy()
    y_prob = prob[test_mask].numpy()
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(classification_report(y_test, y_pred))

    return auc, recall, precision


# 6. Save model to S3
def save_model_to_s3(model):
    print("Saving HeteroGNN model to S3:")
    torch.save(model.state_dict(), "/tmp/gnn_heteroconv.pt")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/gnn_heteroconv.pt", BUCKET_NAME, "models/gnn_heteroconv.pt")
    print(f"  Saved to s3://{BUCKET_NAME}/models/gnn_heteroconv.pt")


if __name__ == "__main__":
    data = load_graph()
    num_nodes = data["transaction"].x.shape[0]
    train_mask, test_mask = build_masks(num_nodes)
    model, x_scaled, y, edge_index_dict, card_ids, device_ids, email_ids = train_model(
        data, train_mask, test_mask)
    evaluate_model(model, x_scaled, y, edge_index_dict, card_ids, device_ids, email_ids, test_mask)
    save_model_to_s3(model)
    print("\nDone!")