import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_models import MultiScaleConceptModel
from models.concept_embedding_network import ConceptEmbeddingNetwork
from models.question_selection_network import QuestionSelectionNetwork


class AdaptiveKnowledgeAssessmentFramework(nn.Module):
    """
    Complete hierarchical probabilistic neural framework integrating
    graph-based concept modeling, Bayesian uncertainty, and adaptive selection.
    """
    def __init__(self, config):
        super(AdaptiveKnowledgeAssessmentFramework, self).__init__()
        self.config = config
        
        # Multi-scale concept model
        self.concept_model = MultiScaleConceptModel(config)
        
        # Concept Embedding Network
        self.cen = ConceptEmbeddingNetwork(config, self.concept_model)
        
        # Question Selection Network
        self.qsn = QuestionSelectionNetwork(config)
        
    def forward(self, question_ids, responses, edge_index, mode='train'):
        """
        Forward pass through the complete framework.
        
        Args:
            question_ids: [batch_size, seq_len]
            responses: [batch_size, seq_len]
            edge_index: Concept graph edges
            mode: 'train' or 'eval'
            
        Returns:
            Dictionary containing all outputs
        """
        # Get multi-scale concept representations
        concept_outputs = self.concept_model(edge_index)
        concept_embeddings = concept_outputs['concept_embeddings']
        
        # Learn knowledge states through CEN
        knowledge_states, mu, sigma = self.cen(
            question_ids, responses, edge_index, concept_embeddings
        )
        
        outputs = {
            'knowledge_states': knowledge_states,
            'mu': mu,
            'sigma': sigma,
            'concept_embeddings': concept_embeddings,
            'pooled_embeddings': concept_outputs['pooled_embeddings'],
            'assignment_matrix': concept_outputs['assignment_matrix']
        }
        
        return outputs
    
    def adaptive_assessment(self, edge_index, max_questions=30, 
                          uncertainty_threshold=0.15):
        """
        Perform adaptive assessment by sequentially selecting questions.
        
        Args:
            edge_index: Concept graph edges
            max_questions: Maximum number of questions to ask
            uncertainty_threshold: Stop when uncertainty below this
            
        Returns:
            selected_questions: List of selected question indices
            responses: List of responses
            final_knowledge_state: Final estimated knowledge
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Initialize
        selected_questions = []
        responses = []
        asked_mask = torch.zeros(1, self.config['num_questions']).to(device)
        
        # Get concept embeddings
        concept_outputs = self.concept_model(edge_index)
        concept_embeddings = concept_outputs['concept_embeddings']
        
        # Initialize knowledge state (uniform prior)
        knowledge_state = torch.zeros(1, self.config['num_concepts']).to(device)
        mu = torch.zeros(1, self.config['num_concepts']).to(device)
        sigma = torch.ones(1, self.config['num_concepts']).to(device)
        
        for step in range(max_questions):
            # Select next question
            with torch.no_grad():
                question_idx, log_prob, value = self.qsn.select_question(
                    knowledge_state, mu, sigma, asked_mask, deterministic=False
                )
            
            selected_questions.append(question_idx.item())
            asked_mask[0, question_idx] = 1
            
            # Simulate or get actual response
            # In practice, this would be provided by the student
            response = self._get_response(question_idx, knowledge_state, concept_embeddings)
            responses.append(response.item())
            
            # Update knowledge state
            q_ids = torch.tensor([selected_questions]).to(device)
            r_ids = torch.tensor([responses]).to(device)
            
            knowledge_state, mu, sigma = self.cen(
                q_ids, r_ids, edge_index, concept_embeddings
            )
            
            # Extract latest state
            knowledge_state = knowledge_state[:, -1, :]
            mu = mu[:, -1, :]
            sigma = sigma[:, -1, :]
            
            # Check stopping criterion
            avg_uncertainty = sigma.mean().item()
            if avg_uncertainty < uncertainty_threshold:
                break
        
        return {
            'selected_questions': selected_questions,
            'responses': responses,
            'final_knowledge_state': knowledge_state,
            'final_mu': mu,
            'final_sigma': sigma,
            'num_questions': len(selected_questions)
        }
    
    def _get_response(self, question_idx, knowledge_state, concept_embeddings):
        """
        Get response (placeholder for actual student response).
        """
        with torch.no_grad():
            prob = self.cen.predict(knowledge_state, question_idx, concept_embeddings)
            response = torch.bernoulli(prob).long()
        return response
    
    def compute_elbo_loss(self, predicted_mu, predicted_sigma, 
                         knowledge_states, prior_mu=0, prior_sigma=1):
        """
        Compute ELBO loss for Bayesian framework (Equation 18).
        
        Args:
            predicted_mu: Predicted mean
            predicted_sigma: Predicted std
            knowledge_states: Knowledge states
            prior_mu: Prior mean
            prior_sigma: Prior std
            
        Returns:
            elbo_loss: Negative ELBO
        """
        # KL divergence between posterior and prior
        kl_div = torch.log(prior_sigma / (predicted_sigma + 1e-8)) + \
                 (predicted_sigma.pow(2) + (predicted_mu - prior_mu).pow(2)) / \
                 (2 * prior_sigma ** 2) - 0.5
        kl_div = kl_div.sum(dim=-1).mean()
        
        # Reconstruction term (simplified)
        reconstruction = -F.mse_loss(predicted_mu, knowledge_states)
        
        # ELBO = reconstruction - KL
        elbo_loss = -reconstruction + kl_div
        
        return elbo_loss
    
    def compute_regularization(self, concept_embeddings, edge_index, 
                              assignment_matrix, policy_probs):
        """
        Compute regularization terms (Equation 18).
        
        Returns:
            Dictionary of regularization losses
        """
        # Graph Laplacian regularization
        adj = torch.zeros(self.config['num_concepts'], self.config['num_concepts'])
        adj[edge_index[0], edge_index[1]] = 1
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj
        laplacian = laplacian.to(concept_embeddings.device)
        
        graph_reg = torch.trace(
            concept_embeddings.t() @ laplacian @ concept_embeddings
        )
        
        # Entropy regularization for policy
        entropy_reg = -(policy_probs * torch.log(policy_probs + 1e-10)).sum(dim=-1).mean()
        
        # Temporal consistency (if applicable)
        # Placeholder - would need temporal knowledge states
        temporal_reg = torch.tensor(0.0).to(concept_embeddings.device)
        
        return {
            'graph_reg': graph_reg,
            'entropy_reg': entropy_reg,
            'temporal_reg': temporal_reg
        }
