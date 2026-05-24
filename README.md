# Fraud Detection with Graph Neural Networks

A full end-to-end machine learning system for real-time fraud detection, built on 590,000 e-commerce transactions. Compares XGBoost against three GNN architectures, deploys the best model as a REST API on AWS Lambda, and serves predictions through a live web dashboard.

**Live Demo:** http://fraud-detection-frontend-selma.s3-website-us-east-1.amazonaws.com

**API Endpoint:** https://oeyo6z6c9d.execute-api.us-east-1.amazonaws.com/prod/predict

---

## Results

| Model | AUC-ROC | Recall | Precision |
|---|---|---|---|
| XGBoost (baseline) | 0.937 | 0.787 | 0.249 |
| GNN: Card edges only | 0.901 | 0.739 | 0.203 |
| GNN: Multi-edge (card + device + email) | 0.897 | 0.747 | 0.195 |
| GNN: HeteroConv | 0.900 | 0.707 | 0.216 |

XGBoost outperforms all GNN variants. The gap is partly explained by the engineered `card_fraud_rate` feature, which manually captures much of the graph signal the GNNs learn automatically.

---

## Architecture

```
Raw Data (IEEE-CIS Kaggle)
│
▼
Feature Engineering          ← features/engineer.py
│  Log transform, card aggregations,
│  fraud rate per card, string columns for graph
▼
Graph Construction           ← graph/build_graph.py
│  Transaction nodes (361 features)
│  Card / Device / Email nodes
│  Heterogeneous edges
▼
Model Training
├── XGBoost baseline         ← models/train_baseline.py
├── GNN card-only            ← models/train_gnn_cardonly.py
├── GNN multi-edge           ← models/train_gnn_multiedge.py
└── GNN HeteroConv           ← models/train_gnn_heteroconv.py
│
▼
Explainability               ← models/explain.py
│  Feature importance by weight and gain
│  Fraud vs legitimate distributions
▼
AWS Lambda API               ← api/lambda_handler.py
│  POST /predict
│  Returns fraud probability + risk level + top features
▼
Web Dashboard                ← frontend/index.html
Deployed on S3 static hosting
```
---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) - provided by Vesta Corporation and the IEEE Computational Intelligence Society.

- 590,540 transactions, 394 features
- 3.5% fraud rate
- Merged transaction + identity tables
- V features (V1-V339) are anonymized behavioral signals from Vesta's payment processing system

---

## Feature Engineering

- Dropped 74 features with >80% missing values
- Log transform on `TransactionAmt`
- Engineered `card_id`, `card_tx_count`, `card_fraud_rate`
- Saved `card_id_str`, `device_str`, `email_str` for graph construction
- 361 final transaction features

---

## Graph Construction

Built a heterogeneous graph with:
- **590,540** transaction nodes (361 features each)
- **14,520** card nodes
- **1,787** device nodes
- **60** email domain nodes
- Edges: transaction → card, transaction → device, transaction → email

---

## GNN Architecture

**Card-only and Multi-edge models (GraphSAGE):**
- Layer 1: 361 → 256 (SAGEConv + ReLU + Dropout 0.3)
- Layer 2: 256 → 128 (SAGEConv + ReLU)
- Classifier: 128 → 2
- Weighted cross-entropy (class weight 27.6x for fraud)
- Early stopping on validation AUC

**HeteroConv model:**
- Separate SAGEConv per edge type (card, device, email) with reverse edges
- Aggregation: sum
- Dropout 0.5, weight decay 1e-4, learning rate 0.0005

---

## Key Findings

- `card_fraud_rate` is the most frequently used feature (1,000+ trees in XGBoost)
- V257, V258, V294 have the highest information gain: Vesta's anonymized behavioral signals are the strongest predictors
- Adding device and email edges to the GNN did not significantly improve results over card-only edges: card-level patterns dominate fraud behavior in this dataset
- HeteroConv improved from AUC 0.892 to 0.900 after adding dropout, weight decay, and lower learning rate

---

## Deployment

**AWS Lambda** — XGBoost model loaded from S3, predictions served via REST API
**API Gateway** — Public POST endpoint at `/predict`
**S3 Static Hosting** — Frontend dashboard

**API Usage:**
```bash
curl -X POST https://oeyo6z6c9d.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 500.0, "ProductCD": "C", "card4": "visa", "card6": "credit", "card_fraud_rate": 0.15, "card_tx_count": 3}'
```

**Response:**
```json
{
  "fraud_probability": 0.7372,
  "risk_level": "High",
  "top_features": [
    {"feature": "card_fraud_rate", "value": 0.15},
    {"feature": "TransactionAmt", "value": 500.0}
  ]
}
```

---

## Project Structure

```
fraud-detection-gnn/
├── features/
│   └── engineer.py              # Feature engineering pipeline
├── graph/
│   └── build_graph.py           # Heterogeneous graph construction
├── models/
│   ├── train_baseline.py        # XGBoost baseline
│   ├── train_gnn_cardonly.py    # GNN with card edges
│   ├── train_gnn_multiedge.py   # GNN with card + device + email edges
│   ├── train_gnn_heteroconv.py  # HeteroConv GNN
│   └── explain.py               # Feature importance plots
├── api/
│   └── lambda_handler.py        # AWS Lambda handler
├── frontend/
│   └── index.html               # Web dashboard
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── outputs/
│   ├── feature_importance_weight.png
│   ├── feature_importance_gain.png
│   └── feature_distributions.png
└── requirements.txt

```

---

## Tech Stack

Python · XGBoost · PyTorch Geometric · AWS Lambda · API Gateway · S3 · scikit-learn · pandas · boto3

---

## Author

**Elif-Selma Yilmaz**
[GitHub](https://github.com/selmayilmaz13) · [LinkedIn](https://www.linkedin.com/in/selma-yilmaz)
