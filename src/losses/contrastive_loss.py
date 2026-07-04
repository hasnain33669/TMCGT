import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossLevelContrastiveLoss(nn.Module):
    """Cross-level contrastive loss: Node-Motif, Motif-Graph, Node-Graph."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.contrast_loss = ContrastiveLoss(temperature)

    def forward(self, z_nodes, z_motifs, z_graphs):
        """Compute cross-level contrastive losses."""
        device = z_nodes.device

        # L_NM: Node ↔ Motif contrastive loss
        if z_nodes.size(0) > 0 and z_motifs.size(0) > 0:
            n_repeat = max(1, z_nodes.size(0) // max(1, z_motifs.size(0)))
            z_motifs_expanded = z_motifs.repeat(n_repeat, 1)[:z_nodes.size(0)]
            loss_nm = self.contrast_loss(z_nodes, z_motifs_expanded)
        else:
            loss_nm = torch.tensor(0.0, device=device)

        # L_MG: Motif ↔ Graph contrastive loss
        if z_motifs.size(0) > 0 and z_graphs.size(0) > 0:
            z_graphs_expanded = z_graphs.repeat(z_motifs.size(0), 1)
            loss_mg = self.contrast_loss(z_motifs, z_graphs_expanded)
        else:
            loss_mg = torch.tensor(0.0, device=device)

        # L_NG: Node ↔ Graph contrastive loss
        if z_nodes.size(0) > 0 and z_graphs.size(0) > 0:
            z_graphs_expanded = z_graphs.repeat(z_nodes.size(0), 1)
            loss_ng = self.contrast_loss(z_nodes, z_graphs_expanded)
        else:
            loss_ng = torch.tensor(0.0, device=device)

        total_loss = loss_nm + loss_mg + loss_ng
        return total_loss, loss_nm, loss_mg, loss_ng


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Handle different batch sizes
        if z1.size(0) != z2.size(0):
            min_size = min(z1.size(0), z2.size(0))
            z1 = z1[:min_size]
            z2 = z2[:min_size]

        if z1.size(0) < 2:
            return torch.tensor(0.0, device=z1.device)

        similarity = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss = F.cross_entropy(similarity, labels)
        return loss
