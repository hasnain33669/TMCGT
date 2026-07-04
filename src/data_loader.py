import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric Data object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 10.0,
                atom.GetFormalCharge() / 5.0,
                float(atom.GetIsAromatic())
            ]
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge indices
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        if edge_index.size(1) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    except Exception:
        return None


def generate_scaffold(smiles, include_chirality=False):
    """Generate the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if include_chirality:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)


def scaffold_split(dataframe, train_frac=0.8, val_frac=0.1, test_frac=0.1, random_state=42):
    """Perform scaffold-based split of the dataset."""
    scaffolds = []
    for _, row in dataframe.iterrows():
        scaffold = generate_scaffold(row['smiles'])
        scaffolds.append(scaffold)

    dataframe = dataframe.copy()
    dataframe['scaffold'] = scaffolds

    # Group molecules by scaffold
    scaffold_groups = defaultdict(list)
    for idx, row in dataframe.iterrows():
        scaffold_groups[row['scaffold']].append(idx)

    # Sort scaffold groups by size
    sorted_scaffolds = sorted(scaffold_groups.keys(),
                              key=lambda s: len(scaffold_groups[s]),
                              reverse=True)

    np.random.seed(random_state)
    shuffled_scaffolds = np.random.permutation(sorted_scaffolds)

    train_indices = []
    val_indices = []
    test_indices = []

    train_count = int(len(dataframe) * train_frac)
    val_count = int(len(dataframe) * val_frac)

    current_count = 0
    current_split = 'train'

    for scaffold in shuffled_scaffolds:
        group_indices = scaffold_groups[scaffold]
        group_size = len(group_indices)

        if current_split == 'train':
            if current_count + group_size <= train_count:
                train_indices.extend(group_indices)
                current_count += group_size
            else:
                current_split = 'val'
                current_count = 0

        if current_split == 'val':
            if current_count + group_size <= val_count:
                val_indices.extend(group_indices)
                current_count += group_size
            else:
                current_split = 'test'
                current_count = 0

        if current_split == 'test':
            test_indices.extend(group_indices)

    remaining = set(range(len(dataframe))) - set(train_indices) - set(val_indices)
    test_indices.extend(remaining)

    train_data = dataframe.iloc[train_indices].drop(columns=['scaffold']).reset_index(drop=True)
    val_data = dataframe.iloc[val_indices].drop(columns=['scaffold']).reset_index(drop=True)
    test_data = dataframe.iloc[test_indices].drop(columns=['scaffold']).reset_index(drop=True)

    return train_data, val_data, test_data


def load_molecule_dataset(data_path, task_type='classification'):
    """Load molecular dataset from CSV file."""
    df = pd.read_csv(data_path)

    # Handle different column formats
    if 'p_np' in df.columns:
        df = df[['smiles', 'p_np']].rename(columns={'p_np': 'value'})
    elif 'class' in df.columns:
        df = df[['smiles', 'class']].rename(columns={'class': 'value'})
    elif 'label' in df.columns:
        df = df[['smiles', 'label']].rename(columns={'label': 'value'})
    else:
        label_col = [c for c in df.columns if 'p_np' in c.lower() or 'label' in c.lower() or 'class' in c.lower()]
        if label_col:
            df = df[['smiles', label_col[0]]].rename(columns={label_col[0]: 'value'})
        else:
            df = df.iloc[:, [0, 1]]
            df.columns = ['smiles', 'value']

    # Clean and validate
    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    # Remove invalid SMILES
    valid_idx = []
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_idx.append(idx)

    df = df.iloc[valid_idx].reset_index(drop=True)

    if task_type == 'classification':
        df['value'] = df['value'].astype(int)

    return df


class MoleculeGraphDataset(InMemoryDataset):
    """PyTorch Geometric dataset for molecular graphs."""
    
    def __init__(self, dataframe, root='data/molecules', transform=None, pre_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        super(MoleculeGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        for idx, row in self.dataframe.iterrows():
            graph = smiles_to_graph(row['smiles'])
            if graph is not None:
                graph.y = torch.tensor([row['value']], dtype=torch.long if row['value'] in [0, 1] else torch.float)
                data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PrecomputedGraphDataset(Dataset):
    """Dataset that precomputes topology and motif features for all graphs."""
    
    def __init__(self, dataset, topo_extractor, motif_extractor, device='cpu'):
        self.dataset = dataset
        self.device = device
        self.topo_features_list = []
        self.motif_matrix_list = []

        for idx in range(len(dataset)):
            data = dataset[idx]
            topo = topo_extractor.extract_features(data)
            motif = motif_extractor.extract_motifs(data)

            self.topo_features_list.append(topo)
            self.motif_matrix_list.append(motif)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        topo = self.topo_features_list[idx]
        motif = self.motif_matrix_list[idx]
        return data, topo, motif


class PrecomputedDataLoader:
    """DataLoader wrapper that yields precomputed features."""
    
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        import numpy as np
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]

            batch_data_list = []
            batch_topo_list = []
            batch_motif_list = []
            batch_labels = []

            for idx in batch_indices:
                data, topo, motif = self.dataset[idx]
                batch_data_list.append(data)
                batch_topo_list.append(topo)
                batch_motif_list.append(motif)
                batch_labels.append(data.y.item())

            batch_data = torch_geometric.data.Batch.from_data_list(batch_data_list)
            batch_topo = torch.cat(batch_topo_list, dim=0)
            batch_motif = torch.cat(batch_motif_list, dim=0)

            yield batch_data, batch_topo, batch_motif, torch.tensor(batch_labels)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
