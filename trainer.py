import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.metrics import compute_metrics
import os


class Trainer:
    """
    Trainer for the Adaptive Knowledge Assessment Framework.
    Implements two-phase training: CEN pre-training + joint optimization.
    """
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizers
        self.cen_optimizer = optim.Adam(
            self.model.cen.parameters(),
            lr=config['training']['learning_rate']
        )
        
        self.qsn_optimizer = optim.Adam(
            self.model.qsn.parameters(),
            lr=config['training']['learning_rate'] * 0.5
        )
        
        # Loss functions
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        
        # Regularization weights
        self.alpha1 = config['regularization']['alpha1_graph']
        self.alpha2 = config['regularization']['alpha2_entropy']
        self.alpha3 = config['regularization']['alpha3_temporal']
        
        # Reward weights
        self.w1 = config['rewards']['w1_accuracy']
        self.w2 = config['rewards']['w2_efficiency']
        self.w3 = config['rewards']['w3_uncertainty_reduction']
        
        # Tracking
        self.best_val_auc = 0
        self.patience_counter = 0
        self.writer = SummaryWriter(config['paths']['log_dir'])
        
    def train_phase1(self, train_loader, val_loader, edge_index, num_epochs):
        """
        Phase 1: Pre-train Concept Embedding Network (Equation 11).
        """
        print("\n=== Phase 1: CEN Pre-training ===")
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                question_ids = batch['question_ids'].to(self.device)
                responses = batch['responses'].to(self.device)
                lengths = batch['lengths']
                
                # Forward pass
                outputs = self.model(question_ids, responses, edge_index, mode='train')
                
                # Compute predictions for each time step
                knowledge_states = outputs['knowledge_states']
                concept_embeddings = outputs['concept_embeddings']
                
                # Response prediction loss
                batch_size, seq_len = question_ids.shape
                loss = 0
                
                for t in range(seq_len - 1):
                    # Predict next response
                    ks_t = knowledge_states[:, t, :]
                    q_next = question_ids[:, t + 1]
                    r_next = responses[:, t + 1].float()
                    
                    # Get prediction
                    pred = self.model.cen.predict(ks_t, q_next, concept_embeddings)
                    
                    # Cross-entropy loss
                    loss += self.ce_loss_fn(pred, r_next)
                
                loss = loss / (seq_len - 1)
                
                # Backward pass
                self.cen_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.cen.parameters(),
                    self.config['training']['grad_clip']
                )
                self.cen_optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader, edge_index)
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, "
                  f"Val ACC: {val_metrics['accuracy']:.4f}")
            
            # Logging
            self.writer.add_scalar('Phase1/train_loss', avg_loss, epoch)
            self.writer.add_scalar('Phase1/val_auc', val_metrics['auc'], epoch)
            
            # Early stopping
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint('best_phase1.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    print("Early stopping triggered")
                    break
        
        self.load_checkpoint('best_phase1.pth')
    
    def train_phase2(self, train_loader, val_loader, edge_index, num_epochs):
        """
        Phase 2: Joint optimization with policy gradient (Equation 12-13).
        """
        print("\n=== Phase 2: Joint Optimization ===")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_reward = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                question_ids = batch['question_ids'].to(self.device)
                responses = batch['responses'].to(self.device)
                
                # Forward pass
                outputs = self.model(question_ids, responses, edge_index, mode='train')
                
                knowledge_states = outputs['knowledge_states']
                mu = outputs['mu']
                sigma = outputs['sigma']
                concept_embeddings = outputs['concept_embeddings']
                
                # Compute losses
                batch_size, seq_len = question_ids.shape
                
                # 1. Response prediction loss
                ce_loss = 0
                for t in range(seq_len - 1):
                    ks_t = knowledge_states[:, t, :]
                    q_next = question_ids[:, t + 1]
                    r_next = responses[:, t + 1].float()
                    pred = self.model.cen.predict(ks_t, q_next, concept_embeddings)
                    ce_loss += self.ce_loss_fn(pred, r_next)
                ce_loss = ce_loss / (seq_len - 1)
                
                # 2. ELBO loss for Bayesian uncertainty
                elbo_loss = self.model.compute_elbo_loss(
                    mu, sigma, knowledge_states
                )
                
                # 3. Policy gradient for question selection
                pg_loss = 0
                rewards = []
                asked_mask = torch.zeros(batch_size, self.config['num_questions']).to(self.device)
                
                for t in range(seq_len):
                    ks_t = knowledge_states[:, t, :]
                    mu_t = mu[:, t, :]
                    sigma_t = sigma[:, t, :]
                    
                    # Get policy distribution
                    _, q_probs, value = self.model.qsn(ks_t, mu_t, sigma_t, asked_mask)
                    
                    # Compute reward (Equation 13)
                    accuracy = (responses[:, t] == 1).float()
                    efficiency = 1.0 / (t + 1)
                    unc_reduction = sigma_t.mean(dim=1) if t > 0 else torch.zeros_like(accuracy)
                    
                    reward = (self.w1 * accuracy + 
                             self.w2 * efficiency + 
                             self.w3 * unc_reduction)
                    
                    rewards.append(reward)
                    
                    # Update asked mask
                    asked_mask[torch.arange(batch_size), question_ids[:, t]] = 1
                
                # Compute policy gradient loss (Equation 12)
                rewards_tensor = torch.stack(rewards, dim=1)
                cumulative_rewards = torch.cumsum(rewards_tensor, dim=1)
                
                for t in range(seq_len):
                    ks_t = knowledge_states[:, t, :]
                    mu_t = mu[:, t, :]
                    sigma_t = sigma[:, t, :]
                    
                    _, q_probs, value = self.model.qsn(ks_t, mu_t, sigma_t)
                    
                    # Select actual question taken
                    q_taken = question_ids[:, t]
                    log_prob = torch.log(q_probs.gather(1, q_taken.unsqueeze(1)) + 1e-10)
                    
                    # Advantage = reward - baseline
                    advantage = cumulative_rewards[:, t] - value.squeeze()
                    
                    pg_loss -= (log_prob.squeeze() * advantage.detach()).mean()
                
                pg_loss = pg_loss / seq_len
                
                # 4. Regularization
                reg_dict = self.model.compute_regularization(
                    concept_embeddings, edge_index,
                    outputs['assignment_matrix'], q_probs
                )
                
                # Total loss (Equation 18)
                total_loss_batch = (ce_loss + elbo_loss + pg_loss +
                                   self.alpha1 * reg_dict['graph_reg'] +
                                   self.alpha2 * reg_dict['entropy_reg'] +
                                   self.alpha3 * reg_dict['temporal_reg'])
                
                # Backward pass
                self.cen_optimizer.zero_grad()
                self.qsn_optimizer.zero_grad()
                total_loss_batch.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                self.cen_optimizer.step()
                self.qsn_optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_reward += rewards_tensor.mean().item()
                
                pbar.set_postfix({
                    'loss': f"{total_loss_batch.item():.4f}",
                    'reward': f"{rewards_tensor.mean().item():.4f}"
                })
            
            avg_loss = total_loss / len(train_loader)
            avg_reward = total_reward / len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader, edge_index)
            
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, Val ATL: {val_metrics['atl']:.2f}")
            
            # Logging
            self.writer.add_scalar('Phase2/train_loss', avg_loss, epoch)
            self.writer.add_scalar('Phase2/train_reward', avg_reward, epoch)
            self.writer.add_scalar('Phase2/val_auc', val_metrics['auc'], epoch)
            self.writer.add_scalar('Phase2/val_atl', val_metrics['atl'], epoch)
            
            # Save best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint('best_phase2.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    print("Early stopping triggered")
                    break
    
    def evaluate(self, data_loader, edge_index):
        """Evaluate model on validation/test set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_test_lengths = []
        
        with torch.no_grad():
            for batch in data_loader:
                question_ids = batch['question_ids'].to(self.device)
                responses = batch['responses'].to(self.device)
                
                outputs = self.model(question_ids, responses, edge_index, mode='eval')
                
                knowledge_states = outputs['knowledge_states']
                concept_embeddings = outputs['concept_embeddings']
                
                # Make predictions
                batch_size, seq_len = question_ids.shape
                for t in range(seq_len - 1):
                    ks_t = knowledge_states[:, t, :]
                    q_next = question_ids[:, t + 1]
                    r_next = responses[:, t + 1]
                    
                    pred = self.model.cen.predict(ks_t, q_next, concept_embeddings)
                    
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(r_next.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = compute_metrics(predictions, targets)
        metrics['atl'] = np.mean([15.0])  # Placeholder
        
        return metrics
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.config['paths']['checkpoint_dir'], filename)
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'cen_optimizer_state_dict': self.cen_optimizer.state_dict(),
            'qsn_optimizer_state_dict': self.qsn_optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
        }, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = os.path.join(self.config['paths']['checkpoint_dir'], filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_auc = checkpoint['best_val_auc']
