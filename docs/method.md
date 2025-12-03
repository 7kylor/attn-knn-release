# Method Description

## Overview

Attn-KNN is an attention-weighted k-nearest neighbor classifier that learns to weight neighbors based on their relevance to the query, rather than using uniform or distance-based weights.

---

## Architecture

### 1. Embedding Network

```
Input Image (32x32x3)
        ↓
    ResNet18/50 (pretrained ImageNet)
        ↓
    Global Average Pooling
        ↓
    Projection Head (Linear → LayerNorm → GELU → Linear)
        ↓
    L2 Normalization
        ↓
    Embedding (256-dim)
```

**Key Design Choices**:

- L2 normalization ensures embeddings lie on unit sphere
- Projection head learns task-specific representation
- Pretrained backbone provides strong initialization

### 2. Memory Bank

```
Training Set Embeddings
        ↓
    FAISS Index (Flat L2 or HNSW)
        ↓
    Top-k Nearest Neighbor Search
        ↓
    k Neighbor Embeddings + Labels
```

**Key Design Choices**:

- FAISS for efficient similarity search
- Memory bank updated periodically during training
- Flat index for datasets <100K, HNSW for larger

### 3. Multi-Head Neighbor Attention

```
Query Embedding (q) ────────┐
                            ↓
Neighbor Embeddings (K) ──→ Attention Scores
                            ↓
                    softmax(Q·K^T / τ)
                            ↓
                    Attention Weights (α)
                            ↓
Neighbor Labels ──────────→ Weighted Vote
                            ↓
                    Class Probabilities
```

**Attention Mechanism**:

```python
# Query projection
Q = W_q @ query  # (1, d_head)

# Key projection
K = W_k @ neighbors  # (k, d_head)

# Attention scores with learned temperature
scores = Q @ K.T / temperature  # (1, k)

# Attention weights
alpha = softmax(scores)  # (1, k)

# Weighted vote
probs = alpha @ one_hot(neighbor_labels)  # (1, num_classes)
```

**Key Design Choices**:

- Per-head learned temperature (τ ∈ [0.05, 2.0])
- Multi-head attention (4 heads) for diverse patterns
- Distance bias adds similarity information to attention

---

## Training Objective

### Combined Loss Function

```python
total_loss = knn_loss + λ_con * contrastive_loss + λ_ent * entropy_loss
```

### 1. kNN Classification Loss

```python
# Attention-weighted neighbor labels
neighbor_onehot = one_hot(neighbor_labels, num_classes)  # (B, k, C)
knn_probs = (attention @ neighbor_onehot)  # (B, C)

# Cross-entropy loss
knn_loss = cross_entropy(log(knn_probs), targets)
```

**Key**: Loss is computed on attention-weighted predictions, aligning training with evaluation.

### 2. Contrastive Loss (Optional)

```python
# Supervised contrastive with margin
sim_matrix = embeddings @ embeddings.T
positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))

# Margin-based contrastive
loss = -log(exp(sim_pos + margin) / sum(exp(sim_all)))
```

### 3. Entropy Regularization

```python
# Encourage diverse attention (prevent collapse)
entropy = -(attention * log(attention)).sum(dim=-1).mean()
entropy_loss = -entropy  # Maximize entropy
```

---

## Evaluation

### Standard Evaluation

```python
# Get k nearest neighbors
neighbors, labels = memory_bank.search(query, k=16)

# Compute attention weights
attention = model.attention(query, neighbors)

# Weighted prediction
probs = attention @ one_hot(labels)
prediction = argmax(probs)
```

### Test-Time Augmentation (TTA)

```python
augmentations = [original, hflip, crop1, crop2, crop3]

all_probs = []
for aug in augmentations:
    aug_query = apply(aug, query)
    probs = predict(aug_query)
    all_probs.append(probs)

final_probs = mean(all_probs)
```

**Key Finding**: TTA provides 78% ECE reduction.

### k-Ensemble

```python
k_values = [5, 10, 20]

all_probs = []
for k in k_values:
    probs = predict(query, k=k)
    all_probs.append(probs)

final_probs = mean(all_probs)
```

**Finding**: k-Ensemble provides no significant improvement.

---

## Hyperparameters

### Recommended Configuration

```yaml
# Model
embed_dim: 256
num_heads: 4
temperature_init: 0.5
temperature_bounds: [0.05, 2.0]

# Training
epochs: 20
batch_size: 512
learning_rate: 2e-4
weight_decay: 1e-4
k_train: 16

# Augmentation
mixup_alpha: 0.4
label_smoothing: 0.1

# Loss weights
contrastive_weight: 0.5
entropy_reg: 0.01

# Evaluation
k_eval: 16
tta_augments: 5
```

---

## Comparison to Baselines

### Uniform kNN

```python
weights = 1/k  # Equal weight for all neighbors
probs = weights @ one_hot(labels)
```

### Distance-Weighted kNN

```python
distances = ||query - neighbors||
weights = softmax(-distances / temperature)
probs = weights @ one_hot(labels)
```

### Attention kNN (Ours)

```python
scores = query @ neighbors.T / learned_temperature
weights = softmax(scores + distance_bias)
probs = weights @ one_hot(labels)
```

**Key Difference**: Learned temperature and optional distance bias.

---

## Computational Complexity

| Component       | Complexity           |
| --------------- | -------------------- |
| Embedding       | O(d) forward pass    |
| kNN Search      | O(k log n) with HNSW |
| Attention       | O(k \* d_head)       |
| Total per query | O(d + k log n)       |

**Practical**: <1ms additional overhead over standard kNN.

---

## Limitations

1. **Accuracy**: Does not significantly beat distance-weighted kNN
2. **Scale**: Memory bank grows with training set size
3. **Embedding Quality**: Performance depends on embedding model
4. **Training**: Requires memory bank updates during training
