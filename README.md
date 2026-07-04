# TMCGT: Topology-Aware Motif Contrastive Graph Transformer for Molecular Property Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TMCGT is a novel deep learning framework for molecular property prediction that integrates:

- **Topology-Aware Learning**: Augments node features with graph-theoretic descriptors (degree centrality, clustering coefficient, PageRank, betweenness centrality)
- **Motif Extraction**: Captures higher-order structural patterns in molecular graphs
- **Multi-Scale GIN Encoding**: Learns expressive local node representations
- **Topology-Aware Graph Transformer**: Models long-range dependencies with structural bias
- **Cross-Level Contrastive Learning**: Aligns node, motif, and graph-level representations
- **Adaptive Feature Fusion**: Dynamically combines local, motif, and global representations

## Key Features

- **Unified Architecture**: End-to-end framework for both classification and regression tasks
- **Comprehensive Evaluation**: Tested on 7 MoleculeNet benchmarks (BBBP, BACE, ClinTox, SIDER, ESOL, FreeSolv, Lipo)
- **State-of-the-Art Performance**: Achieves competitive results across molecular property prediction tasks
- **Modular Design**: Easily extendable components for custom applications

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Install from source

```bash
git clone https://github.com/yourusername/TMCGT-Molecular-Property-Prediction.git
cd TMCGT-Molecular-Property-Prediction
pip install -r requirements.txt
pip install -e .
