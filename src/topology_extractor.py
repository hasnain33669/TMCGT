import torch
import numpy as np
import networkx as nx


class TopologyFeatureExtractor:
    """Extracts comprehensive topology features for each node."""
    
    def __init__(self):
        pass

    def extract_features(self, data):
        """Extract topology features from graph data."""
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return torch.zeros(0, 4)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        # Degree centrality (normalized)
        degrees = np.array([G.degree(i) for i in range(num_nodes)], dtype=np.float32)
        deg_norm = degrees / (degrees.max() + 1e-8)

        # Clustering coefficient
        if G.number_of_edges() > 0:
            clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
        else:
            clustering = np.zeros(num_nodes, dtype=np.float32)

        # PageRank
        try:
            pagerank = np.array(list(nx.pagerank(G).values()), dtype=np.float32)
        except:
            pagerank = np.ones(num_nodes, dtype=np.float32) / num_nodes

        # Betweenness centrality (normalized)
        try:
            betweenness = np.array(list(nx.betweenness_centrality(G).values()), dtype=np.float32)
        except:
            betweenness = np.zeros(num_nodes, dtype=np.float32)

        # Stack all topology features
        features = np.stack([deg_norm, clustering, pagerank, betweenness], axis=1)
        return torch.tensor(features, dtype=torch.float)
