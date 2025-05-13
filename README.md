# GNN From Scratch 🧠🧪  
Implementing Graph Neural Networks for Molecular Machine Learning

This repository contains **from-scratch implementations** of foundational Graph Neural Networks (GNNs) described in the literature, with a focus on molecular property prediction. Each model is implemented in PyTorch, using minimal dependencies and emphasizing clarity and reproducibility.

📍 **Project repo**: [github.com/marcossantanaioc/gnn-from-scratch](https://github.com/marcossantanaioc/gnn-from-scratch)

---

## ✅ Implemented Models

| Model | Paper | Status |
|-------|-------|--------|
| Neural Fingerprints (NFP) | [Duvenaud et al., 2015](https://arxiv.org/abs/1509.09292) | ✅ |
| Message Passing Neural Network (MPNN) | [Gilmer et al., 2017](https://arxiv.org/abs/1704.01212) | ✅ |
| Graph Attention Layer (GAT) | [Veličković et al., 2017](https://arxiv.org/abs/1710.10903) | ✅ |

> ✅ = Implemented

---

## 📁 Repository Structure
gnn-from-scratch/
│
├── models/                    # Core GNN model implementations
│   ├── ngf_model.py           # Neural Fingerprints (NFP)
│   ├── mpnn.py                # Message Passing Neural Network (MPNN)
│
├── layers/                    # GNN building blocks
│   ├── graph_attention_layers.py
│   ├── mpnn_layers.py
│
├── datasets/                 # Dataset loaders and utilities
│   ├── mpnn_dataset.py
│   ├── ngf_dataset.py
│
│
├── tests/                    # Unit tests
│
├── pyproject.toml            # Project dependencies and metadata
├── README.md                 # Project overview and documentation
