#!/usr/bin/env python
"""
Evaluation script for TMCGT model.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from src.data_loader import (
    load_molecule_dataset, scaffold_split, MoleculeGraphDataset,
    PrecomputedGraphDataset, PrecomputedDataLoader
)
from src.topology_extractor import TopologyFeatureExtractor
from src.motif_extractor import MotifExtractor
from src.models.tmcgt import TMCGT
from src.trainer import TMCGTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TMCGT model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='BBBP', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_motifs', type=int, default=8, help='Number of motifs')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split dataset
    data = load_molecule_dataset(args.data_path)
    _, _, test_data = scaffold_split(data, random_state=args.seed)
    
    # Create dataset
    test_dataset = MoleculeGraphDataset(test_data)
    
    # Extract features
    topo_extractor = TopologyFeatureExtractor()
    motif_extractor = MotifExtractor(num_motifs=args.num_motifs)
    
    test_precomputed = PrecomputedGraphDataset(test_dataset, topo_extractor, motif_extractor, device)
    test_loader = PrecomputedDataLoader(test_precomputed, args.batch_size, shuffle=False)
    
    # Load model
    num_classes = len(data['value'].unique())
    model = TMCGT(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        topo_dim=4,
        num_motifs=args.num_motifs,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Evaluate
    trainer = TMCGTTrainer(model, device)
    auc, acc, precision, recall, f1 = trainer.evaluate(test_loader, compute_all_metrics=True)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    main()
