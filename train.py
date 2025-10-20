import torch
import yaml
import argparse
import os
import random
import numpy as np
from models.framework import AdaptiveKnowledgeAssessmentFramework
from trainer import Trainer
from utils.data_utils import (
    load_assistments_data,
    create_concept_graph,
    temporal_split,
    get_data_loaders
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config):
    """Prepare data for training."""
    print("Loading data...")
    
    # Load dataset
    data_path = os.path.join(config['paths']['data_dir'], 'assistments_data.csv')
    df = load_assistments_data(data_path)
    
    print(f"Loaded {len(df)} interactions from {df['student_id'].nunique()} students")
    
    # Get dataset statistics
    num_students = df['student_id'].nunique()
    num_questions = df['question_id'].nunique()
    num_concepts = df['concept_id'].nunique() if 'concept_id' in df.columns else 124
    
    # Update config with dataset info
    config['num_students'] = num_students
    config['num_questions'] = num_questions
    config['num_concepts'] = num_concepts
    
    # Temporal split
    train_df, val_df, test_df = temporal_split(
        df,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_df, val_df, test_df,
        batch_size=config['training']['batch_size'],
        max_seq_len=config['data']['max_sequence_length']
    )
    
    # Create concept graph
    concept_relations_path = os.path.join(
        config['paths']['data_dir'],
        'concept_relations.csv'
    )
    
    if os.path.exists(concept_relations_path):
        edge_index, edge_weight = create_concept_graph(concept_relations_path)
    else:
        print("No concept relations file found, creating random graph...")
        edge_index, edge_weight = create_concept_graph(
            num_concepts=num_concepts
        )
    
    return train_loader, val_loader, test_loader, edge_index, config


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader, edge_index, config = prepare_data(config)
    edge_index = edge_index.to(device)
    
    # Add model config
    model_config = {
        **config['model'],
        'num_concepts': config['num_concepts'],
        'num_questions': config['num_questions'],
        'dropout': config['model']['dropout'],
        'lambda_info': config['question_selection']['lambda_info'],
        'lambda_unc': config['question_selection']['lambda_unc'],
        'lambda_eff': config['question_selection']['lambda_eff'],
    }
    
    # Initialize model
    print("\nInitializing model...")
    model = AdaptiveKnowledgeAssessmentFramework(model_config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Phase 1: CEN Pre-training
    if not args.skip_phase1:
        print("\nStarting Phase 1: CEN Pre-training")
        trainer.train_phase1(
            train_loader,
            val_loader,
            edge_index,
            config['training']['num_epochs_phase1']
        )
    
    # Phase 2: Joint Optimization
    if not args.skip_phase2:
        print("\nStarting Phase 2: Joint Optimization")
        trainer.train_phase2(
            train_loader,
            val_loader,
            edge_index,
            config['training']['num_epochs_phase2']
        )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, edge_index)
    
    print("\n=== Test Results ===")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    if 'atl' in test_metrics:
        print(f"Average Test Length: {test_metrics['atl']:.2f}")
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Adaptive Knowledge Assessment Framework'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--skip_phase1',
        action='store_true',
        help='Skip Phase 1 pre-training'
    )
    parser.add_argument(
        '--skip_phase2',
        action='store_true',
        help='Skip Phase 2 joint optimization'
    )
    
    args = parser.parse_args()
    main(args)
