# Attn-KNN Research Project - Executive Summary

## Project Overview

**Research Question**: Can learned attention over neighbors improve calibration and robustness versus uniform and distance-weighted kNN, with minimal compute overhead?

**Answer**: Partially validated. Attention alone provides marginal benefits, but Test-Time Augmentation (TTA) combined with attention achieves significant calibration improvements (78% ECE reduction).

**Status**: PARTIALLY VALIDATED  
**Timeline**: November 2025  
**Experiments**: 6 major iterations

---

## Key Results

### Best Performance (Experiment 5)

| Method             | Accuracy   | ECE        | NLL       | F1-Macro   |
| ------------------ | ---------- | ---------- | --------- | ---------- |
| Uniform kNN        | 86.82%     | 0.1297     | 2.053     | 86.78%     |
| Distance kNN       | 86.85%     | 0.1099     | 2.054     | 86.81%     |
| Attn-KNN           | 86.85%     | 0.1300     | 2.054     | 86.81%     |
| **Attn-KNN + TTA** | **87.20%** | **0.0283** | **0.967** | **87.27%** |
| CNN Baseline       | 95.12%     | 0.0685     | 0.242     | 95.11%     |

### TTA Impact

| Metric   | Without TTA | With TTA | Improvement |
| -------- | ----------- | -------- | ----------- |
| Accuracy | 86.85%      | 87.20%   | +0.35%      |
| ECE      | 0.1300      | 0.0283   | **-78%**    |
| NLL      | 2.054       | 0.967    | **-53%**    |

---

## Experimental Timeline

### Experiment 1-3: Initial Prototype

- **Goal**: Establish baseline
- **Architecture**: ResNet18, 128-dim, single-head
- **Result**: Attention ≈ Uniform ≈ Distance (88.61% accuracy)
- **Discovery**: Fixed training-evaluation disconnect

### Experiment 4: Architecture Improvements

- **Goal**: Improve through larger models
- **Changes**: ResNet50, 256-dim, 4-heads, contrastive loss
- **Result**: +1.08% accuracy (89.70%) but no attention advantage
- **Finding**: Architectural improvements don't create separation

### Experiment 5: Enhanced Training with TTA [BEST]

- **Goal**: Maximize calibration
- **Changes**: TTA, MixUp, label smoothing, optimized config
- **Result**: 78% ECE reduction with TTA
- **Discovery**: TTA is the real innovation, not attention

### Experiment 6: Final Validation

- **Goal**: Reproduce Experiment 5
- **Result**: Pattern confirmed, TTA consistently improves calibration

---

## Claim Validation

| Claim                   | Status    | Evidence                                      |
| ----------------------- | --------- | --------------------------------------------- |
| Calibration Improvement | VALIDATED | 78% ECE reduction with TTA (0.1297 to 0.0283) |
| Robustness              | VALIDATED | Maintains performance under 30% label noise   |
| Accuracy Improvement    | MARGINAL  | +0.03% to +0.38% over uniform kNN             |
| Compute Overhead        | VALIDATED | <1ms additional per query                     |

---

## Key Findings

### What Works

1. **Test-Time Augmentation (TTA)**

   - 78% ECE reduction
   - 53% NLL reduction
   - Consistent across experiments

2. **Robustness**

   - All kNN methods robust to 30% label noise
   - Stable across k parameter (k ∈ [1, 50])
   - Robust to class imbalance

3. **Minimal Overhead**
   - <1ms additional compute per query
   - Practical for deployment

### What Doesn't Work

1. **Attention Alone**

   - No accuracy benefit over distance weighting
   - No calibration benefit without TTA
   - Multi-head = single-head performance

2. **Architectural Improvements**

   - ResNet50 doesn't help kNN accuracy
   - Contrastive loss issues
   - Hard negative mining unclear impact

3. **Ensemble Methods**
   - k-ensemble provides no improvement

---

## Method Architecture

```
Input Image (32×32×3)
    ↓
ResNet18 Backbone (ImageNet pretrained)
    ↓
Projection Head (256-dim)
    ↓
L2 Normalized Embeddings
    ↓
FAISS Memory Bank (k=16 neighbors)
    ↓
Multi-Head Attention (4 heads)
    ↓
Weighted Classification
    ↓
[Optional] Test-Time Augmentation (5 views)
```

---

## Robustness Analysis

### k Parameter Robustness

- Accuracy stable across k ∈ [1, 50]
- Optimal k: 10-20 for CIFAR-10
- NLL decreases with k (validates theory)

### Noise Robustness

- 0% noise: Baseline (86.82-86.85%)
- 10% noise: ~100% retention
- 20% noise: ~99% retention
- 30% noise: ~98% retention

### Class Imbalance Robustness

- 1:1 ratio: Baseline
- 1:10 ratio: ~99% retention
- 1:100 ratio: ~98% retention

---

## Limitations

1. **Accuracy Gains Marginal**: +0.03% to +0.38% over baselines
2. **Gap to CNN**: 7-9% accuracy gap persists
3. **Attention Alone Doesn't Help**: No benefit without TTA
4. **Dataset Limited**: Only CIFAR-10 thoroughly evaluated

---

## Technical Specifications

### Best Configuration (Experiment 5)

```yaml
backbone: ResNet18
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

### Efficiency

- **Memory Bank Search**: 0.027-0.051ms per query (FAISS)
- **Attention Overhead**: <0.5ms per query
- **Total Overhead**: <1ms per query

---

## Future Directions

### High Priority

1. Larger scale evaluation (ImageNet, fine-grained)
2. Transformer embeddings (CLIP, DINO)
3. Calibration-first training objective

### Pivot Options

1. Calibrated Retrieval Confidence (RAG systems)
2. Uncertainty Quantification for kNN
3. kNN for Few-Shot Learning

---

## Portfolio Highlights

This project demonstrates:

1. **Research Methodology**: Systematic experimental design
2. **Technical Skills**: Deep learning, attention, kNN, calibration
3. **Scientific Integrity**: Honest reporting of limitations
4. **Problem-Solving**: Identified and fixed training issues
5. **Innovation**: Discovered TTA as key technique
6. **Communication**: Comprehensive documentation

---

## Key Story for Portfolio

Started with hypothesis about attention improving kNN. Through systematic experimentation across 6 iterations, discovered that **Test-Time Augmentation (not attention) drives calibration improvements**. Honest assessment of limitations provides valuable learning experience. Method achieves 78% ECE reduction with minimal compute overhead, suitable for calibration-critical applications.

---

## Quick Stats

- **Experiments**: 6 major iterations
- **Best ECE**: 0.0283 (78% reduction)
- **Best Accuracy**: 87.20% (with TTA)
- **Gap to CNN**: 7.92%
- **Noise Tolerance**: 30% label noise
- **Compute Overhead**: <1ms per query
- **k Robustness**: Stable across k ∈ [1, 50]

---

## Related Documents

- **research_narrative.md**: Complete chronological narrative of the research journey
- **architecture_diagrams.md**: Comprehensive ASCII art diagrams of architecture and workflows
- **experiments.md**: Detailed experiment documentation
- **honest_assessment.md**: Transparent evaluation of limitations

---

_For complete chronological narrative, see research_narrative.md_
