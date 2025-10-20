import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj


class HierarchicalGCN(nn.Module):
    """
    Hierarchical Graph Convolutional Network for concept embeddings.
    Implements Equation (1) from the paper.
    """
    def __init__(self, num_concepts, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(HierarchicalGCN, self).__init__()
        self.num_concepts = num_concepts
        self.num_layers = num_layers
        
        # Initial concept embeddings
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, embedding_dim))
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        dims = [embedding_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(dims[i], dims[i+1]))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        
    def forward(self, edge_index, edge_weight=None):
        """
        Forward pass implementing hierarchical graph convolutions.
        
        Args:
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Optional edge weights
            
        Returns:
            Multi-scale concept embeddings
        """
        x = self.concept_embeddings
        embeddings_by_layer = []
        
        for i, conv in enumerate(self.gcn_layers):
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
            embeddings_by_layer.append(x)
        
        return x, embeddings_by_layer


class GraphPooling(nn.Module):
    """
    Differentiable graph pooling for hierarchical concept clustering.
    Implements Equation (2) from the paper.
    """
    def __init__(self, num_concepts, num_clusters, hidden_dim):
        super(GraphPooling, self).__init__()
        self.num_concepts = num_concepts
        self.num_clusters = num_clusters
        
        # Learnable soft assignment matrix S
        self.assignment_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )
        
    def forward(self, concept_embeddings):
        """
        Pool concepts into higher-level clusters.
        
        Args:
            concept_embeddings: [num_concepts, hidden_dim]
            
        Returns:
            pooled_embeddings: [num_clusters, hidden_dim]
            assignment_matrix: [num_concepts, num_clusters]
        """
        # Compute soft assignment matrix with softmax normalization
        assignment_logits = self.assignment_network(concept_embeddings)
        assignment_matrix = F.softmax(assignment_logits, dim=1)  # [K, M]
        
        # Pool embeddings: H^(pool) = S^T H
        pooled_embeddings = torch.matmul(assignment_matrix.t(), concept_embeddings)
        
        return pooled_embeddings, assignment_matrix


class SymmetricGraphConv(nn.Module):
    """
    Symmetric graph convolution with automorphism-invariant properties.
    """
    def __init__(self, in_channels, out_channels):
        super(SymmetricGraphConv, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        G-equivariant graph convolution maintaining symmetry.
        """
        return self.conv(x, edge_index, edge_weight)


class MultiScaleConceptModel(nn.Module):
    """
    Complete graph-based concept modeling with hierarchical structure.
    """
    def __init__(self, config):
        super(MultiScaleConceptModel, self).__init__()
        self.config = config
        
        self.hierarchical_gcn = HierarchicalGCN(
            num_concepts=config['num_concepts'],
            embedding_dim=config['concept_embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_gcn_layers'],
            dropout=config['dropout']
        )
        
        self.graph_pooling = GraphPooling(
            num_concepts=config['num_concepts'],
            num_clusters=config['num_clusters'],
            hidden_dim=config['hidden_dim']
        )
        
    def forward(self, edge_index, edge_weight=None):
        """
        Generate multi-scale concept representations.
        
        Returns:
            concept_embeddings: Fine-grained concept representations
            pooled_embeddings: Coarse-grained cluster representations
            assignment_matrix: Concept-to-cluster assignments
            embeddings_by_layer: Multi-scale representations
        """
        concept_embeddings, embeddings_by_layer = self.hierarchical_gcn(edge_index, edge_weight)
        pooled_embeddings, assignment_matrix = self.graph_pooling(concept_embeddings)
        
        return {
            'concept_embeddings': concept_embeddings,
            'pooled_embeddings': pooled_embeddings,
            'assignment_matrix': assignment_matrix,
            'embeddings_by_layer': embeddings_by_layer
        }
