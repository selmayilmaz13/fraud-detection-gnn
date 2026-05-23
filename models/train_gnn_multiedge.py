"""
Trains a GraphSAGE model using card, device, and email edges.
Transactions sharing the same card, device, or email domain
are connected as neighbors.

Architecture:
- GraphSAGE layer 1: 361 -> 256
- ReLU + Dropout
- GraphSAGE layer 2: 256 -> 128
- ReLU
- Linear classifier: 128 -> 2
- Weighted cross-entropy loss for class imbalance

Steps:
1. Load graph from S3
2. Time-based train/test mask
3. Build transaction-transaction edges via card, device, email
4. Train GraphSAGE model
5. Evaluate with AUC-ROC, Recall, Precision
6. Save model to S3

Output: s3://fraud-detection-gnn/models/gnn_multiedge.pt
"""

import boto3
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report

BUCKET_NAME = "fraud-detection-gnn"
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 150
TRAIN_RATIO = 0.8
MAX_NEIGHBORS = 20


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


# 3. Define GNN model
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden1)
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.classifier = torch.nn.Linear(hidden2, 2)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# 4. Build transaction-transaction edges via card, device, email
def build_edges(data):
    print("Building transaction-transaction edges:")
    tx_src, tx_dst = [], []

    for edge_type in ["card", "device", "email"]:
        edge_index = data["transaction", "uses", edge_type].edge_index
        src = edge_index[0]
        dst = edge_index[1]

        entity_to_transactions = {}
        for tx_idx, entity_idx in zip(src.tolist(), dst.tolist()):
            if entity_idx not in entity_to_transactions:
                entity_to_transactions[entity_idx] = []
            entity_to_transactions[entity_idx].append(tx_idx)

        for entity_idx, tx_list in entity_to_transactions.items():
            if len(tx_list) > MAX_NEIGHBORS:
                tx_list = tx_list[:MAX_NEIGHBORS]
            for i in range(len(tx_list)):
                for j in range(len(tx_list)):
                    if i != j:
                        tx_src.append(tx_list[i])
                        tx_dst.append(tx_list[j])

    tx_edge_index = torch.tensor([tx_src, tx_dst], dtype=torch.long)
    print(f"  Transaction-transaction edges: {tx_edge_index.shape[1]}")
    return tx_edge_index


# 5. Train model
def train_model(data, train_mask, test_mask, tx_edge_index):
    print("Training GNN:")
    x = data["transaction"].x
    y = data["transaction"].y
    scaler = StandardScaler()
    x = torch.tensor(
        scaler.fit_transform(x.numpy()),
        dtype=torch.float)

    n_neg = (y == 0).sum().item()
    n_pos = (y == 1).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)
    print(f"  Class weight for fraud: {n_neg/n_pos:.1f}")

    model = FraudGNN(
        in_channels=x.shape[1],
        hidden1=HIDDEN_DIM_1,
        hidden2=HIDDEN_DIM_2)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_auc = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(x, tx_edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(x, tx_edge_index)
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
    return model, x, y


# 6. Evaluate model
def evaluate_model(model, x, y, tx_edge_index, test_mask):
    print("\nEvaluating GNN:")
    model.eval()
    with torch.no_grad():
        out = model(x, tx_edge_index)
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


# 7. Save model to S3
def save_model_to_s3(model):
    print("Saving model to S3:")
    torch.save(model.state_dict(), "/tmp/gnn_multiedge.pt")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/gnn_multiedge.pt", BUCKET_NAME, "models/gnn_multiedge.pt")
    print(f"  Saved to s3://{BUCKET_NAME}/models/gnn_multiedge.pt")


if __name__ == "__main__":
    data = load_graph()
    num_nodes = data["transaction"].x.shape[0]
    train_mask, test_mask = build_masks(num_nodes)
    tx_edge_index = build_edges(data)
    model, x, y = train_model(data, train_mask, test_mask, tx_edge_index)
    evaluate_model(model, x, y, tx_edge_index, test_mask)
    save_model_to_s3(model)
    print("\nDone!")