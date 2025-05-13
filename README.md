

5_IML_project/
│
├── Edge_pop.ipynb    # Edge-Pop: Active Learning with Subnetwork Sparsity
├── Lent5_LTH.ipynb   # LeNet-5 Structured Pruning with Active Learning


# edge_pop.ipynb
# Edge-Pop: Active Learning with Subnetwork Sparsity

This project implements Edge-Popup subnetworks within a LeNet-5 architecture on the CIFAR-10 dataset using Active Learning. The idea is to learn sparse subnetworks (Edge-Popup networks) that can approximate the performance of the full network, while significantly reducing parameter count and computational cost.

## Overview

- **Dataset**: CIFAR-10  
- **Model**: LeNet-5 (Baseline), Edge-Popup Subnet (Sparse Variant)  
- **Active Learning Strategy**: Least Confidence Sampling  
- **Sparsity Mechanism**: Learned binary masks using "Popup Scores"  
- **Visualization**: Progress and model comparison plots  
- **Evaluation**:
  - Accuracy vs. number of labeled samples
  - Final comparison after full training
  - Sparsity statistics for convolutional layers

## Project Structure

```
edge_pop.ipynb
README.md
/Edge-Pop_0.5_2/
│
├── active_learning_log.csv         # Accuracy log per iteration
├── popup_score_stats.csv           # Stats of popup scores per iteration
├── progress.png                    # Accuracy vs. labeled samples plot
├── model_comparison.png            # Final model accuracy comparison
├── final_subnet.weights.h5         # Subnet model weights
├── final_global_lenet5_full_training.weights.h5
├── global_lenet5.weights.h5        # Pretrained global LeNet-5 weights
└── popup_scores_iter*.npy          # Saved popup scores per layer
```

## How to Run

1. **Dependencies**:
   - TensorFlow
   - NumPy
   - Pandas
   - Matplotlib
   - scikit-learn

2. **Run Notebook**:
   Open `edge_pop.ipynb` in Google Colab or a Jupyter environment. Running the notebook will:
   - Train or load a baseline LeNet-5 model
   - Execute active learning with an Edge-Popup sparse subnetwork
   - Save model weights, results, and plots
   - Retrain both models on the full dataset
   - Compare accuracy and sparsity

## Key Concepts

### Edge-Popup Subnetwork

Edge-Popup learns sparse subnetworks using trainable scores to create binary masks, allowing training of a sparse architecture without pruning a dense model post-hoc.

### Active Learning

This approach selects the most uncertain (least confident) samples to label next, making learning more efficient with fewer labeled samples.

### Model Comparison

Both the global model and the final sparse subnet are evaluated based on:
- Test accuracy
- Number of trainable parameters
- Convolutional layer sparsity

## Example Outputs

**Active Learning Progress**:
Progress plot saved to: `Edge-Pop_0.5_2/progress.png`

**Model Comparison**:
Bar chart saved to: `Edge-Pop_0.5_2/model_comparison.png`

## Future Work

- Test different sparsity ratios (`k` values)
- Explore other sampling strategies (entropy, margin sampling)
- Try with more complex architectures like ResNet or VGG




# Lent5_LTH.ipynb


# LeNet-5 Structured Pruning with Active Learning

## Overview

This project implements **structured pruning** and **active learning** on a LeNet-5 model trained on the CIFAR-10 dataset. It uses iterative pruning based on the Lottery Ticket Hypothesis (LTH), combined with uncertainty-based sample selection to improve training efficiency and model compactness.

The notebook supports full logging, visualization, model saving, and metric tracking across pruning iterations.

---

## Features

- CIFAR-10 image classification using LeNet-5
- Active learning using uncertainty-based sampling
- Structured channel/filter-level pruning
- Iterative retraining and FLOP calculation
- Model summary and history logging
- Metric visualization (accuracy, loss, filters)
- Reinitialization after each pruning stage

---

## Requirements

Make sure the following packages are installed:

```bash
pip install tensorflow numpy matplotlib scikit-learn tensorflow-model-optimization
```

Other built-in modules used:

- `os`, `json`, `csv`, `datetime`

---

## Directory Structure

The script will generate logs and results in a timestamped folder:

```
training_logs_<timestamp>/
│
├── graphs/           # Plots of accuracy, loss, and filters
├── metrics/          # Training metrics CSV and history JSON
├── models/           # Saved model checkpoints and summaries
└── experiment_summary.txt
```

---

## How It Works

1. **Data Preparation**
   - CIFAR-10 dataset is loaded and preprocessed.
   - Initial labeled and unlabeled splits are created.

2. **Model Definition**
   - A LeNet-5 architecture is defined and compiled.
   - A clone of the base model is kept for reinitialization after pruning.

3. **Active Learning + Pruning Cycle**
   - Train the model on labeled data.
   - Evaluate and log metrics including accuracy, loss, and FLOPs.
   - Select the most uncertain samples and add to the labeled set.
   - Apply structured pruning to reduce parameters.
   - Reinitialize model while retaining the pruned structure.

4. **Logging and Visualization**
   - Accuracy and loss per iteration
   - Number of filters kept after each pruning stage
   - Full training history and model summaries

---

## Hyperparameters

Default values used in this experiment:

```json
{
  "batch_size": 128,
  "learning_rate": 0.001,
  "num_classes": 10,
  "initial_sample_size": 2000,
  "budget_per_iteration": 6000,
  "iterations": 5,
  "num_epochs": 20,
  "prune_ratio": 0.02
}
```

---

## Output

- **Model files**: Saved after each iteration
- **Metric logs**: `training_metrics.csv` (includes FLOPs, accuracy, loss)
- **History**: Accuracy/loss/filter history in JSON
- **Visualizations**: Automatically saved plots for trends over pruning

---

## How to Run

1. Launch Jupyter and open the notebook:

```bash
jupyter notebook Lent5_LTH_Active.ipynb
```

2. Execute the notebook cells sequentially to run the full pipeline.

---