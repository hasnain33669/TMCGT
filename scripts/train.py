#!/usr/bin/env python
"""
Training script for TMCGT model.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data_loader import (
    load_molecule_dataset, scaffold_split, MoleculeGraphDataset,
    PrecomputedGraphDataset, PrecomputedDataLoader
)
from src.topology_extractor import TopologyFeatureExtractor
from src.motif_extractor import MotifExtractor
from src.models.tmcgt import TMCGT
from src.trainer import TMCGTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TMCGT model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--dataset_name', type=str, default='BBBP', help='Dataset name')
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_motifs', type=int, default=8, help='Number of motifs')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    data = load_molecule_dataset(args.data_path, args.task_type)
    
    # Split dataset
    train_data, val_data, test_data = scaffold_split(data, random_state=args.seed)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = MoleculeGraphDataset(train_data)
    val_dataset = MoleculeGraphDataset(val_data)
    test_dataset = MoleculeGraphDataset(test_data)
    
    # Extract features
    topo_extractor = TopologyFeatureExtractor()
    motif_extractor = MotifExtractor(num_motifs=args.num_motifs)
    
    train_precomputed = PrecomputedGraphDataset(train_dataset, topo_extractor, motif_extractor, device)
    val_precomputed = PrecomputedGraphDataset(val_dataset, topo_extractor, motif_extractor, device)
    test_precomputed = PrecomputedGraphDataset(test_dataset, topo_extractor, motif_extractor, device)
    
    # Create dataloaders
    train_loader = PrecomputedDataLoader(train_precomputed, args.batch_size, shuffle=True)
    val_loader = PrecomputedDataLoader(val_precomputed, args.batch_size, shuffle=False)
    test_loader = PrecomputedDataLoader(test_precomputed, args.batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(data['value'].unique()) if args.task_type == 'classification' else 1
    model = TMCGT(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        topo_dim=4,
        num_motifs=args.num_motifs,
        num_classes=num_classes,
        dropout=args.dropout,
        lambda_contrast=0.001,
        lambda_topo=0.05,
        lambda_motif=0.05
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainer = TMCGTTrainer(model, device)
    
    # Training loop
    best_auc = 0
    best_acc = 0
    
    print("\n" + "=" * 80)
    print(f"{'Epoch':<8} {'Train_Loss':<12} {'Train_Acc':<10} {'Val_AUC':<10} {'Val_Acc':<10}")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
        
        # Validation
        val_auc, val_acc = trainer.evaluate(val_loader)
        
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.save_dir}/best_model.pt')
        
        print(f"{epoch+1:<8} {train_loss:<12.6f} {train_acc:<10.4f} {val_auc:<10.6f} {val_acc:<10.4f}")
    
    print("=" * 80)
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_auc, test_acc, test_precision, test_recall, test_f1 = trainer.evaluate(
        test_loader, compute_all_metrics=True
    )
    
    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
