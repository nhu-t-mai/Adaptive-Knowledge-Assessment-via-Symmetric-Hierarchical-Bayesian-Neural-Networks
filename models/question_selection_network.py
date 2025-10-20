import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PermutationEquivariantAttention(nn.Module):
    """
    Permutation-equivariant attention mechanism.
    Implements Equations (5) and (6) from the paper.
    """
    def __init__(self, hidden_dim, num_concepts):
        super(PermutationEquivariantAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Shared parameter matrices for equivariance (Equation 6)
        self.W_h = nn.Linear(num_concepts, hidden_dim)
        self.W_u = nn.Linear(num_concepts, hidden_dim)
        self.w_a = nn.Linear(hidden_dim, 1)
        
    def forward(self, knowledge_state, uncertainty_state):
        """
        Compute attention weights maintaining permutation equivariance.
        
        Args:
            knowledge_state: [batch, num_concepts]
            uncertainty_state: [batch, num_concepts]
            
        Returns:
            attention_weights: [batch, num_concepts]
        """
        # Transform states
        h_transformed = self.W_h(knowledge_state)
        u_transformed = self.W_u(uncertainty_state)
        
        # Combine with tanh activation
        combined = torch.tanh(h_transformed + u_transformed)
        
        # Compute attention logits
        attention_logits = self.w_a(combined).squeeze(-1)
        
        # Softmax normalization (Equation 6)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        return attention_weights


class QuestionSelectionNetwork(nn.Module):
    """
    Question Selection Network (QSN) using policy gradient methods.
    Implements information-theoretic selection with uncertainty awareness.
    """
    def __init__(self, config):
        super(QuestionSelectionNetwork, self).__init__()
        self.config = config
        self.num_questions = config['num_questions']
        self.num_concepts = config['num_concepts']
        self.hidden_dim = config['hidden_dim']
        
        # Question-concept mapping
        self.question_concept_mapping = nn.Parameter(
            torch.randn(config['num_questions'], config['num_concepts'])
        )
        
        # Attention mechanism
        self.attention = PermutationEquivariantAttention(
            self.hidden_dim,
            self.num_concepts
        )
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(self.num_concepts * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_questions)
        )
        
        # Value network (baseline for variance reduction)
        self.value_network = nn.Sequential(
            nn.Linear(self.num_concepts * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Hyperparameters from config
        self.lambda_info = config.get('lambda_info', 0.7)
        self.lambda_unc = config.get('lambda_unc', 0.2)
        self.lambda_eff = config.get('lambda_eff', 0.1)
        
    def compute_information_gain(self, knowledge_state, question_idx, mu, sigma):
        """
        Compute mutual information gain (Equation 9).
        
        Args:
            knowledge_state: [batch, num_concepts]
            question_idx: [batch] or scalar
            mu: Mean of knowledge distribution
            sigma: Std of knowledge distribution
            
        Returns:
            information_gain: Expected information gain
        """
        batch_size = knowledge_state.shape[0]
        
        # Current entropy H(hs)
        entropy_current = 0.5 * torch.log(2 * np.pi * np.e * sigma.pow(2)).sum(dim=-1)
        
        # Expected conditional entropy after observing response
        # Approximation: assume response reduces uncertainty
        q_concepts = F.softmax(self.question_concept_mapping[question_idx], dim=-1)
        
        # Reduced sigma for relevant concepts
        sigma_reduced = sigma * (1 - 0.5 * q_concepts)
        entropy_conditional = 0.5 * torch.log(2 * np.pi * np.e * sigma_reduced.pow(2)).sum(dim=-1)
        
        # Information gain
        information_gain = entropy_current - entropy_conditional
        
        return information_gain
    
    def compute_uncertainty_score(self, sigma, question_idx):
        """
        Compute uncertainty score for question's relevant concepts.
        """
        q_concepts = F.softmax(self.question_concept_mapping[question_idx], dim=-1)
        uncertainty_score = (sigma * q_concepts).sum(dim=-1)
        return uncertainty_score
    
    def compute_question_score(self, knowledge_state, mu, sigma, question_idx, 
                               question_costs=None):
        """
        Compute question selection score (Equation 10).
        
        Args:
            knowledge_state: Current knowledge state
            mu: Mean of knowledge distribution
            sigma: Std of knowledge distribution
            question_idx: Question indices to score
            question_costs: Optional costs for each question
            
        Returns:
            scores: Selection scores for questions
        """
        # Information gain component
        info_gain = self.compute_information_gain(knowledge_state, question_idx, mu, sigma)
        
        # Uncertainty component
        unc_score = self.compute_uncertainty_score(sigma, question_idx)
        
        # Efficiency component (cost)
        if question_costs is None:
            cost = torch.ones_like(info_gain)
        else:
            cost = question_costs[question_idx]
        
        # Combined score (Equation 10)
        scores = (self.lambda_info * info_gain + 
                 self.lambda_unc * unc_score - 
                 self.lambda_eff * cost)
        
        return scores
    
    def forward(self, knowledge_state, mu, sigma, asked_questions_mask=None):
        """
        Select next question using policy network.
        
        Args:
            knowledge_state: [batch, num_concepts]
            mu: [batch, num_concepts]
            sigma: [batch, num_concepts]
            asked_questions_mask: [batch, num_questions] - 1 for asked, 0 for not asked
            
        Returns:
            question_logits: [batch, num_questions]
            question_probs: [batch, num_questions]
            value: [batch, 1] baseline value
        """
        batch_size = knowledge_state.shape[0]
        
        # Compute attention weights
        attention_weights = self.attention(knowledge_state, sigma)
        
        # Weighted representation
        attended_knowledge = knowledge_state * attention_weights
        
        # Concatenate features for policy
        policy_input = torch.cat([knowledge_state, mu, sigma], dim=-1)
        
        # Compute question logits
        question_logits = self.policy_network(policy_input)
        
        # Mask already asked questions
        if asked_questions_mask is not None:
            question_logits = question_logits.masked_fill(
                asked_questions_mask.bool(), float('-inf')
            )
        
        # Question probabilities
        question_probs = F.softmax(question_logits, dim=-1)
        
        # Compute baseline value
        value_input = torch.cat([knowledge_state, sigma], dim=-1)
        value = self.value_network(value_input)
        
        return question_logits, question_probs, value
    
    def select_question(self, knowledge_state, mu, sigma, asked_questions_mask=None,
                       deterministic=False):
        """
        Select a question (for inference).
        
        Args:
            knowledge_state: Knowledge state
            mu: Mean
            sigma: Std
            asked_questions_mask: Mask of asked questions
            deterministic: If True, select argmax; else sample
            
        Returns:
            selected_question: Question index
            log_prob: Log probability of selection
        """
        question_logits, question_probs, value = self.forward(
            knowledge_state, mu, sigma, asked_questions_mask
        )
        
        if deterministic:
            selected_question = torch.argmax(question_probs, dim=-1)
        else:
            # Sample from policy distribution
            dist = torch.distributions.Categorical(probs=question_probs)
            selected_question = dist.sample()
        
        # Compute log probability
        log_prob = torch.log(question_probs.gather(1, selected_question.unsqueeze(1)) + 1e-10)
        
        return selected_question, log_prob, value
