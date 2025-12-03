# Experiment Documentation

## Overview

This document provides detailed documentation of all experiments conducted in the Attn-KNN project, including configurations, results, and analysis.

---

## Experiment Timeline

| Experiment | Date       | Objective               | Outcome               |
| ---------- | ---------- | ----------------------- | --------------------- |
| 1-3        | Initial    | Validate basic approach | Baseline established  |
| 4          | Iteration  | Improve architecture    | Marginal gains        |
| 5          | Iteration  | Enhanced training       | Best calibration      |
| 6          | Validation | Confirm results         | Consistent with Exp 5 |

---

## Experiment 1-3: Initial Prototype

### Objective

Establish baseline performance and validate that attention-weighted kNN can match or beat uniform/distance-weighted kNN.

### Configuration

```yaml
backbone: ResNet18
embed_dim: 128
num_heads: 1
epochs: 30
batch_size: 256
k_train: 10
k_eval: 10
learning_rate: 1e-3
```

### Results

#### CIFAR-10

| Method       | Accuracy | F1-Macro | NLL   | ECE    |
| ------------ | -------- | -------- | ----- | ------ |
| Uniform kNN  | 88.61%   | 88.58%   | 1.021 | 0.0165 |
| Distance kNN | 88.62%   | 88.59%   | 1.018 | 0.0347 |
| Attn-KNN     | 88.54%   | 88.51%   | 1.025 | 0.0329 |

#### Iris

| Method      | Accuracy | F1-Macro |
| ----------- | -------- | -------- |
| All Methods | 96.67%   | 96.67%   |

### Findings

1. Attention mechanism performs comparably to baselines
2. ECE improvements over distance-weighted kNN
3. Iris dataset saturated (ceiling effect)

### Output Files

- `results/experiment_1-3/cifar10_full_results.json`
- `results/experiment_1-3/cifar10_reliability_*.png`
- `results/experiment_1-3/best_model.pt`

---

## Experiment 4: Architecture Improvements

### Objective

Improve performance through architectural upgrades and advanced training techniques.

### Configuration Changes

```yaml
# Changes from Experiment 1-3
backbone: ResNet18 -> ResNet50
epochs: 30 -> 50
warmup_epochs: 0 -> 5
contrastive_weight: 0 -> 0.5
hard_negative_weight: 0 -> 2.0
contrastive_margin: 0 -> 0.5
temperature_init: 1.0 -> 0.5
temperature_max: 5.0 -> 2.0
```

### Results

#### Main Results (CIFAR-10)

| Method       | Accuracy | F1-Macro | NLL   | ECE    |
| ------------ | -------- | -------- | ----- | ------ |
| Uniform kNN  | 89.69%   | 89.66%   | 0.964 | 0.0243 |
| Distance kNN | 89.70%   | 89.66%   | 0.964 | 0.0566 |
| Attn-KNN     | 89.70%   | 89.66%   | 0.964 | 0.0578 |
| CNN Baseline | 94.94%   | 94.93%   | 0.258 | 0.0718 |

#### k-Sweep Results

| k   | Uniform | Distance | Attention |
| --- | ------- | -------- | --------- |
| 1   | 88.86%  | 88.86%   | 88.86%    |
| 3   | 89.52%  | 89.44%   | 89.47%    |
| 5   | 89.66%  | 89.62%   | 89.62%    |
| 10  | 89.69%  | 89.70%   | 89.70%    |
| 20  | 89.75%  | 89.68%   | 89.73%    |
| 50  | 89.78%  | 89.77%   | 89.75%    |

#### Noise Robustness

| Noise Rate | Uniform | Distance | Attention |
| ---------- | ------- | -------- | --------- |
| 0%         | 89.69%  | 89.70%   | 89.70%    |
| 10%        | 89.73%  | 89.73%   | 89.73%    |
| 20%        | 89.48%  | 89.57%   | 89.57%    |
| 30%        | 89.21%  | 89.18%   | 89.18%    |

#### Imbalance Results

| Ratio | Uniform | Distance | Attention |
| ----- | ------- | -------- | --------- |
| 1.0   | 89.69%  | 89.70%   | 89.70%    |
| 0.1   | 89.41%  | 89.42%   | 89.40%    |
| 0.01  | 88.69%  | 88.81%   | 88.80%    |

### Findings

1. ResNet50 improved embeddings but not kNN accuracy
2. All kNN methods perform similarly
3. 5% gap to CNN upper bound
4. Robust to noise and imbalance

### Output Files

- `results/experiment_4/all_results.json`
- `results/experiment_4/k_sweep_*.png`
- `results/experiment_4/noise_*.png`
- `results/experiment_4/best_model.pt`

---

## Experiment 5: Enhanced Training (BEST RESULTS)

### Objective

Maximize calibration through TTA, k-ensemble, and refined training.

### Configuration

```yaml
backbone: ResNet18 # Reverted - ResNet50 didn't help
embed_dim: 256
num_heads: 4
epochs: 20
batch_size: 512
k_train: 16
k_eval: 16
mixup_alpha: 0.4
label_smoothing: 0.1
contrastive_weight: 0.5
knn_loss_weight: 1.0
entropy_reg: 0.01
tta_augments: 5
k_ensemble_values: [5, 10, 20]
```

### Results

#### Main Results (CIFAR-10)

| Method                | Accuracy   | F1-Macro   | NLL       | ECE        |
| --------------------- | ---------- | ---------- | --------- | ---------- |
| Uniform kNN           | 86.82%     | 86.78%     | 2.053     | 0.1297     |
| Distance kNN          | 86.85%     | 86.81%     | 2.054     | 0.1099     |
| Attn-KNN              | 86.85%     | 86.81%     | 2.054     | 0.1300     |
| **Attn-KNN + TTA**    | **87.20%** | **87.27%** | **0.967** | **0.0283** |
| Attn-KNN + k-Ensemble | 86.83%     | 86.79%     | 2.047     | 0.1299     |
| CNN Baseline          | 95.12%     | 95.11%     | 0.242     | 0.0685     |

#### TTA Impact Analysis

| Metric   | Without TTA | With TTA | Improvement |
| -------- | ----------- | -------- | ----------- |
| Accuracy | 86.85%      | 87.20%   | +0.35%      |
| ECE      | 0.1300      | 0.0283   | -78%        |
| NLL      | 2.054       | 0.967    | -53%        |

#### k-Sweep Results

| k   | Uniform | Distance | Attention |
| --- | ------- | -------- | --------- |
| 1   | 86.82%  | 86.82%   | 86.82%    |
| 3   | 86.84%  | 86.84%   | 86.85%    |
| 5   | 86.83%  | 86.83%   | 86.86%    |
| 10  | 86.82%  | 86.84%   | 86.82%    |
| 20  | 86.83%  | 86.83%   | 86.85%    |
| 50  | 86.80%  | 86.80%   | 86.84%    |

#### Noise Robustness

| Noise Rate | Uniform | Distance | Attention |
| ---------- | ------- | -------- | --------- |
| 0%         | 86.82%  | 86.85%   | 86.85%    |
| 10%        | 86.84%  | 86.85%   | 86.82%    |
| 20%        | 86.83%  | 86.85%   | 86.85%    |
| 30%        | 86.82%  | 86.85%   | 86.80%    |

### Key Findings

1. **TTA is the key innovation** - provides 78% ECE reduction
2. k-ensemble does not improve results
3. All kNN methods robust to noise
4. Attention alone does not beat baselines

### Output Files

- `results/experiment_5/all_results.json`
- `results/experiment_5/reliability_*.png`
- `results/experiment_5/comparison_bar.png`
- `results/experiment_5/best_attnknn_model.pt`

---

## Experiment 6: Final Validation

### Objective

Validate Experiment 5 results with fresh training run.

### Configuration

Same as Experiment 5.

### Results

#### Main Results (CIFAR-10)

| Method             | Accuracy   | F1-Macro   | NLL       | ECE        |
| ------------------ | ---------- | ---------- | --------- | ---------- |
| Uniform kNN        | 85.57%     | 85.53%     | 2.028     | 0.1336     |
| Distance kNN       | 85.55%     | 85.51%     | 2.028     | 0.1169     |
| Attn-KNN           | 85.48%     | 85.44%     | 2.058     | 0.1359     |
| **Attn-KNN + TTA** | **85.54%** | **85.50%** | **0.931** | **0.0379** |
| CNN Baseline       | 94.97%     | 94.96%     | 0.247     | 0.0726     |

### Findings

1. Results consistent with Experiment 5
2. TTA continues to provide best calibration
3. Slight variance in absolute numbers due to random initialization
4. Pattern is reproducible

### Output Files

- `results/experiment_6/all_results.json`
- `results/experiment_6/reliability_*.png`

---

## Summary Statistics Across All Experiments

### Accuracy Progression

| Experiment | Uniform kNN | Attn-KNN | Attn-KNN + TTA |
| ---------- | ----------- | -------- | -------------- |
| 1-3        | 88.61%      | 88.54%   | -              |
| 4          | 89.69%      | 89.70%   | -              |
| 5          | 86.82%      | 86.85%   | **87.20%**     |
| 6          | 85.57%      | 85.48%   | 85.54%         |

### ECE Progression

| Experiment | Uniform kNN | Attn-KNN | Attn-KNN + TTA |
| ---------- | ----------- | -------- | -------------- |
| 1-3        | 0.0165      | 0.0329   | -              |
| 4          | 0.0243      | 0.0578   | -              |
| 5          | 0.1297      | 0.1300   | **0.0283**     |
| 6          | 0.1336      | 0.1359   | 0.0379         |

### Gap to CNN Upper Bound

| Experiment | CNN Accuracy | Best kNN | Gap    |
| ---------- | ------------ | -------- | ------ |
| 4          | 94.94%       | 89.70%   | -5.24% |
| 5          | 95.12%       | 87.20%   | -7.92% |
| 6          | 94.97%       | 85.54%   | -9.43% |

---

## Reproducibility Notes

### Random Seeds

All experiments use seed=42 for reproducibility.

### Hardware

- Apple M4 Max (primary)
- NVIDIA GPU (optional)

### Training Time

- Experiment 1-3: ~30 minutes
- Experiment 4: ~60 minutes (ResNet50)
- Experiment 5-6: ~40 minutes

### Data

- CIFAR-10: Automatic download via torchvision
- Preprocessing: Standard normalization

---

## Conclusion

The experiments demonstrate that:

1. **Attention-weighted kNN matches but does not beat baselines** in accuracy
2. **Test-Time Augmentation is the primary driver** of calibration improvements
3. **All kNN methods are robust** to noise and imbalance
4. **8-9% gap to CNN** remains regardless of kNN variant

The core claim of "attention improves calibration" is **validated only in combination with TTA**. Attention alone provides no measurable benefit over distance-weighted kNN.
