import torch
import numpy as np
import networkx as nx


class MotifExtractor:
    """Extracts multiple motif patterns from the graph."""
    
    def __init__(self, num_motifs=8):
        self.num_motifs = num_motifs

    def extract_motifs(self, data):
        """Extract motif patterns from graph data."""
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return torch.zeros(0, self.num_motifs)

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        motif_matrix = torch.zeros(num_nodes, self.num_motifs)

        if G.number_of_edges() == 0:
            return motif_matrix

        # Motif 0: Triangles (3-cliques)
        triangles = nx.triangles(G)
        for node in G.nodes():
            motif_matrix[node, 0] = triangles[node]

        # Motif 1: 4-cycles (squares)
        for cycle in nx.simple_cycles(G.to_directed()):
            if len(cycle) == 4:
                for node in cycle:
                    motif_matrix[node, 1] += 1

        # Motif 2: 5-cycles
        for cycle in nx.simple_cycles(G.to_directed()):
            if len(cycle) == 5:
                for node in cycle:
                    motif_matrix[node, 2] += 1

        # Motif 3: Star patterns (high-degree nodes)
        for node in G.nodes():
            deg = G.degree(node)
            if deg >= 3:
                motif_matrix[node, 3] = deg

        # Motif 4: Paths of length 2
        for node in G.nodes():
            motif_matrix[node, 4] = G.degree(node)

        # Motif 5: Claw patterns
        for node in G.nodes():
            if G.degree(node) >= 3:
                leaf_count = sum(1 for neighbor in G.neighbors(node) if G.degree(neighbor) == 1)
                motif_matrix[node, 5] = leaf_count

        # Motif 6: Diamond pattern (K4 minus one edge)
        for clique in nx.find_cliques(G):
            if len(clique) == 4:
                for node in clique:
                    motif_matrix[node, 6] += 1

        # Motif 7: Lollipop (cycle with a path)
        cycles = list(nx.cycle_basis(G))
        for cycle in cycles:
            for node in cycle:
                if any(nbr not in cycle for nbr in G.neighbors(node)):
                    motif_matrix[node, 7] += 1

        # Normalize each motif
        for m in range(self.num_motifs):
            max_val = motif_matrix[:, m].max()
            if max_val > 0:
                motif_matrix[:, m] /= max_val

        return motif_matrix
