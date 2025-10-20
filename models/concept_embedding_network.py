import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_models import SymmetricGraphConv


class ConceptEmbeddingNetwork(nn.Module):
    """
    Concept Embedding Network (CEN) that learns student-specific knowledge states.
    Implements Equations (3) and (4) from the paper.
    """
    def __init__(self, config, concept_model):
        super(ConceptEmbeddingNetwork, self).__init__()
        self.config = config
        self.concept_model = concept_model
        self.num_concepts = config['num_concepts']
        self.hidden_dim = config['hidden_dim']
        self.lstm_hidden = config['lstm_hidden_dim']
        
        # Response embedding layer
        self.response_embedding = nn.Embedding(2, config['concept_embedding_dim'])  # 0: incorrect, 1: correct
        
        # Question embedding
        self.question_embedding = nn.Embedding(
            config['num_questions'], 
            config['concept_embedding_dim']
        )
        
        # Interaction encoder
        self.interaction_encoder = nn.Sequential(
            nn.Linear(config['concept_embedding_dim'] * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # BiLSTM for temporal modeling (Equation 3)
        self.bilstm = nn.LSTM(
            self.hidden_dim,
            self.lstm_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout']
        )
        
        # Graph convolution for knowledge state update (Equation 4)
        self.graph_conv = SymmetricGraphConv(
            self.lstm_hidden * 2,  # bidirectional
            self.hidden_dim
        )
        
        # Knowledge state projection
        self.knowledge_state_projection = nn.Linear(
            self.hidden_dim,
            self.num_concepts
        )
        
        # Bayesian uncertainty parameters
        self.mu_head = nn.Linear(self.num_concepts, self.num_concepts)
        self.sigma_head = nn.Sequential(
            nn.Linear(self.num_concepts, self.num_concepts),
            nn.Softplus()  # Ensure positive sigma
        )
        
    def forward(self, question_ids, responses, edge_index, concept_embeddings):
        """
        Forward pass through CEN.
        
        Args:
            question_ids: [batch_size, seq_len]
            responses: [batch_size, seq_len]
            edge_index: Graph edge indices
            concept_embeddings: Pre-computed concept embeddings
            
        Returns:
            knowledge_states: [batch_size, seq_len, num_concepts]
            mu: Mean of knowledge distribution
            sigma: Std of knowledge distribution
        """
        batch_size, seq_len = question_ids.shape
        
        # Encode interactions
        q_emb = self.question_embedding(question_ids)
        r_emb = self.response_embedding(responses)
        
        # Concatenate question and response embeddings
        interaction_emb = torch.cat([q_emb, r_emb], dim=-1)
        encoded = self.interaction_encoder(interaction_emb)
        
        # BiLSTM temporal modeling (Equation 3)
        lstm_out, _ = self.bilstm(encoded)  # [batch, seq_len, lstm_hidden*2]
        
        # Graph convolution for each time step
        knowledge_states_list = []
        for t in range(seq_len):
            # Extract features at time t for all batch samples
            ft = lstm_out[:, t, :]  # [batch, lstm_hidden*2]
            
            # Expand to concept level
            ft_expanded = ft.unsqueeze(1).expand(-1, self.num_concepts, -1)
            ft_flat = ft_expanded.reshape(-1, self.lstm_hidden * 2)
            
            # Create batch edge index
            batch_edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
            batch_edge_index = batch_edge_index.reshape(2, -1)
            
            # Apply graph convolution (Equation 4)
            hs_t = self.graph_conv(ft_flat, batch_edge_index)
            hs_t = hs_t.reshape(batch_size, self.num_concepts, -1)
            
            # Project to knowledge state
            hs_t = self.knowledge_state_projection(hs_t.mean(dim=1))
            knowledge_states_list.append(hs_t)
        
        knowledge_states = torch.stack(knowledge_states_list, dim=1)
        
        # Bayesian uncertainty (Equation 8)
        mu = self.mu_head(knowledge_states)
        sigma = self.sigma_head(knowledge_states)
        
        return knowledge_states, mu, sigma
    
    def predict(self, knowledge_state, question_id, concept_embeddings):
        """
        Predict response probability for a question given knowledge state.
        
        Args:
            knowledge_state: [batch_size, num_concepts]
            question_id: [batch_size]
            concept_embeddings: Concept representations
            
        Returns:
            prob: Predicted correctness probability
        """
        q_emb = self.question_embedding(question_id)
        
        # Compute relevance to each concept
        relevance = torch.matmul(q_emb, concept_embeddings.t())
        relevance = F.softmax(relevance, dim=-1)
        
        # Weighted knowledge state
        weighted_knowledge = (knowledge_state * relevance).sum(dim=-1)
        
        # Predict probability
        prob = torch.sigmoid(weighted_knowledge)
        
        return prob
