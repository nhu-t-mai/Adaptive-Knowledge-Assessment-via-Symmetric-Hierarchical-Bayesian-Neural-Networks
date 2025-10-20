import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import torch


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted probabilities [N]
        targets: True labels [N]
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Remove padding (zeros)
    mask = targets >= 0
    predictions = predictions[mask]
    targets = targets[mask]
    
    # AUC
    try:
        auc = roc_auc_score(targets, predictions)
    except:
        auc = 0.5
    
    # Accuracy
    pred_labels = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(targets, pred_labels)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'rmse': rmse
    }


def compute_calibration_error(predictions, targets, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        predictions: Predicted probabilities
        targets: True labels
        n_bins: Number of bins
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_information_gain(prior_entropy, posterior_entropy):
    """
    Compute information gain.
    
    Args:
        prior_entropy: Entropy before question
        posterior_entropy: Entropy after question
        
    Returns:
        Information gain
    """
    return prior_entropy - posterior_entropy


def compute_uncertainty(sigma):
    """
    Compute uncertainty from standard deviation.
    
    Args:
        sigma: Standard deviation tensor
        
    Returns:
        Average uncertainty
    """
    if isinstance(sigma, torch.Tensor):
        return sigma.mean().item()
    return np.mean(sigma)


class MetricsTracker:
    """Track metrics over training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.test_lengths = []
        self.uncertainties = []
    
    def update(self, predictions, targets, test_length=None, uncertainty=None):
        """Update with new predictions."""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if test_length is not None:
            self.test_lengths.append(test_length)
        if uncertainty is not None:
            self.uncertainties.append(uncertainty)
    
    def compute(self):
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = compute_metrics(predictions, targets)
        
        if len(self.test_lengths) > 0:
            metrics['atl'] = np.mean(self.test_lengths)
            metrics['std_atl'] = np.std(self.test_lengths)
        
        if len(self.uncertainties) > 0:
            metrics['avg_uncertainty'] = np.mean(self.uncertainties)
        
        # Calibration
        metrics['ece'] = compute_calibration_error(predictions, targets)
        
        return metrics
