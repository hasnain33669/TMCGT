import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GINEncoder(nn.Module):
    """GIN Encoder with topology features."""
    
    def __init__(self, in_dim=4, hidden_dim=256, topo_dim=4, num_layers=3, dropout=0.2):
        super().__init__()

        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.topo_proj = nn.Linear(topo_dim, hidden_dim)

        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, topo_features):
        x = self.node_proj(x)
        if topo_features is not None and topo_features.size(0) == x.size(0):
            x = x + self.topo_proj(topo_features)

        for i, (gin, bn) in enumerate(zip(self.gin_layers, self.bn_layers)):
            x = gin(x, edge_index)
            x = bn(x)
            if i < len(self.gin_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x
