import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifConsistencyLoss(nn.Module):
    """Ensure motif representations remain consistent across the graph."""
    
    def __init__(self):
        super().__init__()

    def forward(self, motif_matrix, node_embeddings):
        """Compute motif consistency loss."""
        device = node_embeddings.device

        if motif_matrix.size(0) == 0 or node_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=device)

        # Compute motif prototypes as weighted average of node embeddings
        motif_prototypes = []
        for m in range(motif_matrix.size(1)):
            weights = motif_matrix[:, m]
            if weights.sum() > 0:
                weights = weights / (weights.sum() + 1e-8)
                prototype = (weights.unsqueeze(1) * node_embeddings).sum(dim=0)
            else:
                prototype = torch.zeros(node_embeddings.size(1), device=device)
            motif_prototypes.append(prototype)

        motif_prototypes = torch.stack(motif_prototypes, dim=0)

        # Compute consistency: nodes belonging to same motif should have similar embeddings
        motif_assignments = motif_matrix > 0.5
        loss = 0.0
        count = 0

        for m in range(motif_matrix.size(1)):
            nodes_in_motif = motif_assignments[:, m].nonzero(as_tuple=True)[0]
            if len(nodes_in_motif) > 1:
                motif_embs = node_embeddings[nodes_in_motif]
                motif_embs_norm = F.normalize(motif_embs, dim=-1)
                sim_matrix = torch.mm(motif_embs_norm, motif_embs_norm.t())
                loss += (1 - sim_matrix).mean()
                count += 1

        if count > 0:
            loss = loss / count

        return loss
