# Adaptive Knowledge Assessment via Symmetric Hierarchical Bayesian Neural Networks

Implementation of the paper "Adaptive Knowledge Assessment via Symmetric Hierarchical Bayesian Neural Networks with Graph Symmetry-Aware Concept Dependencies".

## Overview

This repository provides a complete implementation of a hierarchical probabilistic neural framework that integrates Bayesian inference with symmetric deep neural architectures for adaptive, efficient knowledge assessment. The system models student knowledge as latent representations within a graph-structured concept dependency network.

## Key Features

- **Graph-Based Concept Modeling**: Hierarchical GCN with multi-scale concept representations
- **Bayesian Uncertainty Quantification**: Variational inference with symmetric parameterization
- **Adaptive Question Selection**: Information-theoretic selection with reinforcement learning
- **Symmetric Neural Architectures**: Automorphism-invariant embeddings and equivariant transformations
- **Two-Phase Training**: CEN pre-training followed by joint optimization

## Performance

- **76.3% diagnostic accuracy** on ASSISTments dataset
- **35.1% reduction** in required questions compared to traditional assessments
- **Superior calibration** (ECE = 0.048)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/XXX/adaptive-knowledge-assessment.git
cd adaptive-knowledge-assessment

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── config/
│   └── config.yaml              # Configuration file
├── models/
│   ├── graph_models.py          # Graph-based concept modeling
│   ├── concept_embedding_network.py  # CEN implementation
│   ├── question_selection_network.py # QSN implementation
│   └── framework.py             # Complete framework
├── utils/
│   ├── data_utils.py            # Data loading and preprocessing
│   └── metrics.py               # Evaluation metrics
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── trainer.py                   # Trainer class
└── requirements.txt             # Dependencies
```

## Data Preparation

### Expected Data Format

The system expects interaction data in CSV format with the following columns:

- `student_id`: Unique student identifier
- `question_id`: Unique question identifier
- `response`: Binary response (0 = incorrect, 1 = correct)
- `timestamp`: (Optional) Timestamp for temporal ordering
- `concept_id`: (Optional) Associated concept

### Concept Dependency Graph

Optionally provide a concept relations file with:

- `source`: Source concept ID
- `target`: Target concept ID (prerequisite relationship)
- `weight`: (Optional) Edge weight

Place data files in the `data/` directory:

```
data/
├── assistments_data.csv
└── concept_relations.csv  (optional)
```

## Usage

### Training

Basic training with default configuration:

```bash
python train.py --config config/config.yaml
```

Training options:

```bash
python train.py \
    --config config/config.yaml \
    --batch_size 64 \
    --learning_rate 0.001 \
    --seed 42
```

Skip phases:

```bash
# Skip Phase 1 (CEN pre-training)
python train.py --skip_phase1

# Skip Phase 2 (joint optimization)
python train.py --skip_phase2
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --config config/config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --eval_adaptive \
    --save_plots
```

This will:
- Evaluate knowledge tracing performance (AUC, accuracy, RMSE, ECE)
- Run adaptive assessment simulations
- Generate evaluation plots

## Configuration

Key configuration parameters in `config/config.yaml`:

### Model Parameters

```yaml
model:
  concept_embedding_dim: 128
  hidden_dim: 256
  num_gcn_layers: 3
  num_attention_heads: 4
  dropout: 0.2
  num_clusters: 50
```

### Training Parameters

```yaml
training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs_phase1: 100  # CEN pre-training
  num_epochs_phase2: 200  # Joint optimization
```

### Question Selection

```yaml
question_selection:
  lambda_info: 0.7      # Information gain weight
  lambda_unc: 0.2       # Uncertainty weight
  lambda_eff: 0.1       # Efficiency weight
  max_test_length: 30
  uncertainty_threshold: 0.15
```

## Model Architecture

### 1. Graph-Based Concept Modeling

Hierarchical GCN layers capture multi-scale concept dependencies:

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

### 2. Concept Embedding Network (CEN)

- BiLSTM for temporal modeling
- Graph convolutions for knowledge state updates
- Bayesian uncertainty quantification

### 3. Question Selection Network (QSN)

- Permutation-equivariant attention
- Policy gradient optimization
- Information-theoretic scoring

### 4. Training Objectives

Total loss combines:
- Cross-entropy for response prediction
- ELBO for Bayesian uncertainty
- Policy gradient for question selection
- Regularization terms (graph, entropy, temporal)

## Results

### ASSISTments Dataset

| Method | AUC | Accuracy | ATL |
|--------|-----|----------|-----|
| MI | 0.621 | - | 23.4 |
| KLI | 0.635 | - | 22.1 |
| BKT | 0.651 | - | 21.5 |
| DKT | 0.684 | - | 19.8 |
| SAKT | 0.712 | - | 18.2 |
| GKT | 0.726 | - | 17.6 |
| SDKT | 0.739 | - | 16.8 |
| **Ours** | **0.763** | **-** | **15.2** |

### Key Improvements

- **2.4% AUC improvement** over best baseline (SDKT)
- **9.5% reduction in test length** (1.6 fewer questions on average)
- **Superior calibration**: ECE = 0.048 vs. 0.068 for SDKT

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Nhu Tam Mai: ntmai@usc.edu
- Wenyang Cao: wenyangc@usc.edu
