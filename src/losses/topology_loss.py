import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyPreservationLoss(nn.Module):
    """Preserve graph topology in the embedding space."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, node_embeddings, edge_index):
        """Compute topology preservation loss."""
        device = node_embeddings.device
        num_nodes = node_embeddings.size(0)

        if num_nodes < 2 or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=device)

        # Compute pairwise distances in embedding space
        emb_norm = F.normalize(node_embeddings, dim=-1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        dist_matrix = 1 - sim_matrix

        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        # Connected nodes should be close, unconnected nodes should be far
        pos_loss = (adj * dist_matrix).sum() / (adj.sum() + 1e-8)

        # Negative pairs: nodes that are not connected
        neg_adj = 1 - adj
        neg_adj.fill_diagonal_(0)
        neg_loss = torch.relu(self.margin - dist_matrix) * neg_adj
        neg_loss = neg_loss.sum() / (neg_adj.sum() + 1e-8)

        loss = pos_loss + neg_loss
        return loss
