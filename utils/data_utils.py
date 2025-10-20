import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp


class EducationalDataset(Dataset):
    """
    Dataset for educational assessment data.
    """
    def __init__(self, interactions, max_seq_len=200):
        """
        Args:
            interactions: DataFrame with columns [student_id, question_id, response, concept_ids]
            max_seq_len: Maximum sequence length
        """
        self.max_seq_len = max_seq_len
        self.interactions = interactions
        
        # Group by student
        self.student_sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create sequences for each student."""
        sequences = []
        
        for student_id, group in self.interactions.groupby('student_id'):
            group = group.sort_values('timestamp') if 'timestamp' in group.columns else group
            
            question_ids = group['question_id'].values
            responses = group['response'].values
            
            # Split into chunks if sequence is too long
            for i in range(0, len(question_ids), self.max_seq_len):
                seq_q = question_ids[i:i+self.max_seq_len]
                seq_r = responses[i:i+self.max_seq_len]
                
                if len(seq_q) > 1:  # Need at least 2 interactions
                    sequences.append({
                        'question_ids': seq_q,
                        'responses': seq_r,
                        'length': len(seq_q)
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.student_sequences)
    
    def __getitem__(self, idx):
        seq = self.student_sequences[idx]
        
        question_ids = torch.LongTensor(seq['question_ids'])
        responses = torch.LongTensor(seq['responses'])
        length = seq['length']
        
        # Pad if necessary
        if length < self.max_seq_len:
            pad_len = self.max_seq_len - length
            question_ids = torch.cat([
                question_ids,
                torch.zeros(pad_len, dtype=torch.long)
            ])
            responses = torch.cat([
                responses,
                torch.zeros(pad_len, dtype=torch.long)
            ])
        
        return {
            'question_ids': question_ids,
            'responses': responses,
            'length': length
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    question_ids = torch.stack([item['question_ids'] for item in batch])
    responses = torch.stack([item['responses'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    
    return {
        'question_ids': question_ids,
        'responses': responses,
        'lengths': lengths
    }


def load_assistments_data(data_path):
    """
    Load ASSISTments dataset.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with processed interactions
    """
    df = pd.read_csv(data_path)
    
    # Rename columns to standard format
    column_mapping = {
        'user_id': 'student_id',
        'skill_id': 'concept_id',
        'problem_id': 'question_id',
        'correct': 'response'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['student_id', 'question_id', 'response']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert to binary response
    df['response'] = (df['response'] > 0).astype(int)
    
    return df


def create_concept_graph(concept_relations_path=None, num_concepts=None):
    """
    Create concept dependency graph.
    
    Args:
        concept_relations_path: Path to concept relations CSV (optional)
        num_concepts: Number of concepts (required if no relations file)
        
    Returns:
        edge_index: [2, num_edges] tensor
        edge_weight: Optional edge weights
    """
    if concept_relations_path:
        # Load from file
        df = pd.read_csv(concept_relations_path)
        edges = df[['source', 'target']].values
        edge_index = torch.LongTensor(edges.T)
        
        edge_weight = None
        if 'weight' in df.columns:
            edge_weight = torch.FloatTensor(df['weight'].values)
    else:
        # Create random graph structure (for demonstration)
        if num_concepts is None:
            raise ValueError("Must provide num_concepts if no relations file")
        
        # Create sparse random graph
        density = 0.1
        adj_matrix = sp.random(num_concepts, num_concepts, density=density, format='coo')
        edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
    
    return edge_index, edge_weight


def temporal_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data temporally.
    
    Args:
        df: DataFrame with interactions
        train_ratio, val_ratio, test_ratio: Split ratios
        
    Returns:
        train_df, val_df, test_df
    """
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


def get_data_loaders(train_df, val_df, test_df, batch_size=64, max_seq_len=200):
    """
    Create data loaders for training.
    
    Args:
        train_df, val_df, test_df: DataFrames
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EducationalDataset(train_df, max_seq_len)
    val_dataset = EducationalDataset(val_df, max_seq_len)
    test_dataset = EducationalDataset(test_df, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader
