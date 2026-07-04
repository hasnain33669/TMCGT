# TMCGT: Topology-aware Multi-level Contrastive Graph Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

**TMCGT** (Topology-aware Multi-level Contrastive Graph Transformer) is a versatile deep learning framework for molecular property prediction that supports:

- ✅ **Binary Classification** (e.g., BBBP, Tox21)
- ✅ **Multi-class Classification** (e.g., Mutagenicity)
- ✅ **Regression** (e.g., ESOL, FreeSolv, Lipophilicity)

The model combines:
- **Graph Isomorphism Networks (GIN)** for local feature extraction
- **Graph Transformers** for global context modeling
- **Multi-level contrastive learning** at node, motif, and graph levels
- **Topology-aware features** (degree, clustering coefficient, PageRank, betweenness centrality)
- **Motif extraction** for capturing structural patterns

## 🚀 Key Features

- **Multi-level Contrastive Learning**: Node-Motif, Motif-Graph, and Node-Graph contrastive losses
- **Topology Preservation**: Ensures structural information is preserved in embeddings
- **Motif Consistency**: Maintains consistency within motif representations
- **Flexible Architecture**: Easily adaptable for binary, multi-class, and regression tasks
- **Multiple Dataset Support**: Works with any molecular dataset (BBBP, Tox21, ESOL, FreeSolv, etc.)
- **Scaffold-based Splitting**: Uses Bemis-Murcko scaffolds for robust train/val/test splits

## 📊 Supported Tasks

| Task Type 
|-----------
| Binary Classification 
| Multi-class Classification 
| Regression 

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/tmcgt.git
cd tmcgt
pip install -r requirements.txt
pip install -e .
