# Changelog

All notable changes to the Attn-KNN project are documented here.

## Version History

### v0.6 - Final Validation (Experiment 6)

**Date**: November 2025

**Changes**:

- Validation run of Experiment 5 configuration
- Confirmed reproducibility of results
- Minor documentation updates

**Results**:

- Attn-KNN + TTA: 85.54% accuracy, 0.0379 ECE
- Consistent with Experiment 5 patterns

---

### v0.5 - Enhanced Training (Experiment 5) [BEST RESULTS]

**Date**: November 2025

**Major Changes**:

- Increased embedding dimension: 128 -> 256
- Multi-head attention: 1 -> 4 heads
- Added Test-Time Augmentation (TTA)
- Added k-Ensemble method
- Added MixUp augmentation (alpha=0.4)
- Added label smoothing (0.1)

**Configuration**:

```yaml
embed_dim: 256
num_heads: 4
epochs: 20
batch_size: 512
k_train: 16
k_eval: 16
mixup_alpha: 0.4
label_smoothing: 0.1
tta_augments: 5
```

**Results**:

- Best calibration achieved: ECE = 0.0283 (with TTA)
- 78% ECE improvement over baseline
- Accuracy: 87.20% (with TTA)

**Key Finding**: TTA is the primary driver of calibration improvements.

---

### v0.4 - Architecture Improvements (Experiment 4)

**Date**: November 2025

**Major Changes**:

- Upgraded backbone: ResNet18 -> ResNet50
- Increased epochs: 30 -> 50
- Added margin-based contrastive loss
- Added hard negative mining
- Added temperature bounds [0.05, 2.0]
- Added warmup epochs (5)

**Configuration**:

```yaml
backbone: ResNet50
epochs: 50
warmup_epochs: 5
contrastive_weight: 0.5
hard_negative_weight: 2.0
contrastive_margin: 0.5
temperature_init: 0.5
```

**Results**:

- Accuracy: 89.70% (attention)
- Marginal improvement over baselines
- Added noise robustness tests
- Added class imbalance tests

**Finding**: Architectural improvements stabilized training but did not significantly improve accuracy.

---

### v0.3 - Training Objective Fix (Experiment 3)

**Date**: November 2025

**Major Changes**:

- Fixed training-evaluation disconnect
- Loss now computed on attention-weighted neighbor aggregation
- Aligned training objective with evaluation

**Configuration**:

```python
# Before (broken)
logits = classifier(query_embedding)
loss = cross_entropy(logits, labels)

# After (fixed)
neighbor_probs = attention @ neighbor_labels
loss = cross_entropy(log(neighbor_probs), labels)
```

**Results**:

- Training now converges properly
- Evaluation matches training objective

---

### v0.2 - Initial Baselines (Experiment 2)

**Date**: November 2025

**Major Changes**:

- Added distance-weighted kNN baseline
- Added CNN upper bound
- Implemented k-sweep experiments
- Added reliability diagrams

**Results**:

- Established baseline comparisons
- Identified ~5-9% gap to CNN

---

### v0.1 - Initial Prototype (Experiment 1)

**Date**: November 2025

**Initial Implementation**:

- ResNet18 backbone
- Single-head attention
- FAISS memory bank
- Basic kNN classification

**Results**:

- Initial accuracy: ~88% on CIFAR-10
- Validated basic approach

---

## Migration Notes

### From v0.4 to v0.5

- Reverted to ResNet18 (ResNet50 didn't help)
- Reduced epochs (50 -> 20)
- Added TTA for evaluation

### From v0.3 to v0.4

- Major architecture changes
- New loss functions
- Extended training

---

## Known Issues

### Resolved

- Training-evaluation disconnect (fixed in v0.3)
- Contrastive loss outputting 0 (fixed in v0.4)
- Memory bank staleness (fixed with frequent updates)

### Unresolved

- 8-9% gap to CNN upper bound
- Attention doesn't beat distance-weighted kNN in accuracy
- High ECE without TTA

---

## Deprecations

### Removed Features

- Single-head attention (replaced by multi-head)
- Fixed temperature (replaced by learned temperature)
- ResNet50 backbone (reverted to ResNet18)

### Deprecated Configurations

- `embed_dim: 128` - Use 256 instead
- `num_heads: 1` - Use 4 instead
- `epochs: 50` - Use 20 with TTA instead
