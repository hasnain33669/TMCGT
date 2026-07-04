import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer with topology bias."""
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def compute_shortest_path_bias(self, edge_index, num_nodes):
        """Compute shortest path distances as attention bias."""
        device = edge_index.device
        
        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        # Compute shortest paths using BFS from each node
        sp_dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
        sp_dist[range(num_nodes), range(num_nodes)] = 0

        for i in range(num_nodes):
            visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            queue = [i]
            visited[i] = True
            dist = 0
            while queue and dist < num_nodes:
                next_queue = []
                for node in queue:
                    sp_dist[i, node] = dist
                    neighbors = (adj[node] == 1).nonzero(as_tuple=True)[0]
                    for nbr in neighbors:
                        if not visited[nbr]:
                            visited[nbr] = True
                            next_queue.append(nbr)
                queue = next_queue
                dist += 1

        # Convert to bias: -inf for unreachable, -distance for reachable
        bias = -sp_dist
        bias[sp_dist == float('inf')] = -1e9
        return bias

    def forward(self, x, edge_index=None):
        batch_size, seq_len, dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Compute attention scores
        attn = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale

        # Add topology bias if edge_index provided
        if edge_index is not None:
            bias = self.compute_shortest_path_bias(edge_index[0], seq_len)
            attn = attn + bias.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(batch_size, seq_len, dim)
        out = self.proj(out)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x
