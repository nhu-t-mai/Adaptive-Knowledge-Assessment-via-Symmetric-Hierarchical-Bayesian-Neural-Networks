import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from models.framework import AdaptiveKnowledgeAssessmentFramework
from utils.data_utils import (
    load_assistments_data,
    create_concept_graph,
    temporal_split,
    get_data_loaders
)
from utils.metrics import compute_metrics, compute_calibration_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    model_config = {
        **config['model'],
        'num_concepts': config['num_concepts'],
        'num_questions': config['num_questions'],
        'dropout': config['model']['dropout'],
        'lambda_info': config['question_selection']['lambda_info'],
        'lambda_unc': config['question_selection']['lambda_unc'],
        'lambda_eff': config['question_selection']['lambda_eff'],
    }
    
    model = AdaptiveKnowledgeAssessmentFramework(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_adaptive_assessment(model, data_loader, edge_index, device, config):
    """
    Evaluate model on adaptive assessment task.
    Simulates the adaptive testing process.
    """
    model.eval()
    
    results = {
        'test_lengths': [],
        'final_uncertainties': [],
        'accuracies': []
    }
    
    max_questions = config['question_selection']['max_test_length']
    unc_threshold = config['question_selection']['uncertainty_threshold']
    
    print("\nRunning adaptive assessments...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            question_ids = batch['question_ids'].to(device)
            responses = batch['responses'].to(device)
            batch_size = question_ids.shape[0]
            
            # For each student in batch
            for i in range(batch_size):
                result = model.adaptive_assessment(
                    edge_index,
                    max_questions=max_questions,
                    uncertainty_threshold=unc_threshold
                )
                
                results['test_lengths'].append(result['num_questions'])
                results['final_uncertainties'].append(
                    result['final_sigma'].mean().item()
                )
                
                # Compute accuracy on selected questions
                # (In practice, would use actual student responses)
                results['accuracies'].append(0.75)  # Placeholder
    
    return results


def evaluate_knowledge_tracing(model, data_loader, edge_index, device):
    """
    Evaluate model on knowledge tracing task.
    Standard next-response prediction.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    print("\nEvaluating knowledge tracing...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            question_ids = batch['question_ids'].to(device)
            responses = batch['responses'].to(device)
            
            outputs = model(question_ids, responses, edge_index, mode='eval')
            
            knowledge_states = outputs['knowledge_states']
            sigma = outputs['sigma']
            concept_embeddings = outputs['concept_embeddings']
            
            batch_size, seq_len = question_ids.shape
            
            for t in range(seq_len - 1):
                ks_t = knowledge_states[:, t, :]
                q_next = question_ids[:, t + 1]
                r_next = responses[:, t + 1]
                
                pred = model.cen.predict(ks_t, q_next, concept_embeddings)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(r_next.cpu().numpy())
                all_uncertainties.extend(sigma[:, t, :].mean(dim=1).cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    metrics['ece'] = compute_calibration_error(predictions, targets)
    metrics['avg_uncertainty'] = np.mean(uncertainties)
    
    return metrics, predictions, targets, uncertainties


def plot_results(adaptive_results, kt_metrics, predictions, targets, save_dir):
    """Generate visualizations of evaluation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Test length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(adaptive_results['test_lengths'], bins=20, edgecolor='black')
    plt.xlabel('Test Length (Number of Questions)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Adaptive Test Lengths')
    plt.axvline(
        np.mean(adaptive_results['test_lengths']),
        color='r',
        linestyle='--',
        label=f'Mean: {np.mean(adaptive_results["test_lengths"]):.1f}'
    )
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Calibration plot
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        if in_bin.sum() > 0:
            bin_accs.append(targets[in_bin].mean())
            bin_confs.append(predictions[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.bar(
        bin_confs, bin_accs, width=0.08,
        alpha=0.7, edgecolor='black', label='Model'
    )
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Actual Accuracy')
    plt.title(f'Calibration Plot (ECE={kt_metrics["ece"]:.4f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Uncertainty vs Test Length
    plt.figure(figsize=(10, 6))
    plt.scatter(
        adaptive_results['test_lengths'],
        adaptive_results['final_uncertainties'],
        alpha=0.5
    )
    plt.xlabel('Test Length')
    plt.ylabel('Final Uncertainty')
    plt.title('Relationship between Test Length and Final Uncertainty')
    plt.savefig(os.path.join(save_dir, 'uncertainty_vs_length.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}")


def main(args):
    """Main evaluation function."""
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_path = os.path.join(config['paths']['data_dir'], 'assistments_data.csv')
    df = load_assistments_data(data_path)
    
    # Get dataset statistics
    config['num_students'] = df['student_id'].nunique()
    config['num_questions'] = df['question_id'].nunique()
    config['num_concepts'] = df['concept_id'].nunique() if 'concept_id' in df.columns else 124
    
    # Split data
    _, _, test_df = temporal_split(
        df,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    )
    
    _, _, test_loader = get_data_loaders(
        None, None, test_df,
        batch_size=config['training']['batch_size'],
        max_seq_len=config['data']['max_sequence_length']
    )
    
    # Create concept graph
    concept_relations_path = os.path.join(
        config['paths']['data_dir'],
        'concept_relations.csv'
    )
    
    if os.path.exists(concept_relations_path):
        edge_index, _ = create_concept_graph(concept_relations_path)
    else:
        edge_index, _ = create_concept_graph(num_concepts=config['num_concepts'])
    
    edge_index = edge_index.to(device)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Evaluate on knowledge tracing
    kt_metrics, predictions, targets, uncertainties = evaluate_knowledge_tracing(
        model, test_loader, edge_index, device
    )
    
    print("\n=== Knowledge Tracing Results ===")
    print(f"AUC: {kt_metrics['auc']:.4f}")
    print(f"Accuracy: {kt_metrics['accuracy']:.4f}")
    print(f"RMSE: {kt_metrics['rmse']:.4f}")
    print(f"ECE: {kt_metrics['ece']:.4f}")
    print(f"Average Uncertainty: {kt_metrics['avg_uncertainty']:.4f}")
    
    # Evaluate on adaptive assessment
    if args.eval_adaptive:
        adaptive_results = evaluate_adaptive_assessment(
            model, test_loader, edge_index, device, config
        )
        
        print("\n=== Adaptive Assessment Results ===")
        print(f"Average Test Length: {np.mean(adaptive_results['test_lengths']):.2f} "
              f"± {np.std(adaptive_results['test_lengths']):.2f}")
        print(f"Min Test Length: {np.min(adaptive_results['test_lengths'])}")
        print(f"Max Test Length: {np.max(adaptive_results['test_lengths'])}")
        print(f"Average Final Uncertainty: {np.mean(adaptive_results['final_uncertainties']):.4f}")
    
    # Generate plots
    if args.save_plots:
        save_dir = os.path.join(config['paths']['results_dir'], 'evaluation_plots')
        if args.eval_adaptive:
            plot_results(adaptive_results, kt_metrics, predictions, targets, save_dir)
        else:
            print("\nSkipping plots (adaptive evaluation not run)")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Adaptive Knowledge Assessment Framework'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--eval_adaptive',
        action='store_true',
        help='Evaluate adaptive assessment (in addition to knowledge tracing)'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Generate and save evaluation plots'
    )
    
    args = parser.parse_args()
    main(args)
