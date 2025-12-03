# Attn-KNN Research Project - Executive Summary

## Project Overview

**Research Question**: Can learned attention over neighbors improve calibration and robustness versus uniform and distance-weighted kNN, with minimal compute overhead?

**Answer**: NO. Attention alone provides no meaningful benefits. Test-Time Augmentation (TTA) achieves calibration improvements, but this is independent of attention.

**Status**: CONCLUDED - No further investigation needed  
**Timeline**: November - December 2025  
**Experiments**: 7 major iterations

---

## Key Results

### Final Performance (Experiment 7 - December 2025)

| Method             | Accuracy   | ECE        | NLL       |
| ------------------ | ---------- | ---------- | --------- |
| Uniform kNN        | 91.53%     | 0.0796     | 1.225     |
| Distance kNN       | 91.52%     | 0.0783     | 1.225     |
| Attn-KNN           | 91.55%     | 0.0811     | 1.236     |
| **Attn-KNN + TTA** | 90.99%     | **0.0267** | **0.513** |
| CNN Baseline       | **96.51%** | 0.0253     | 0.184     |

### TTA Impact

| Metric   | Without TTA | With TTA | Improvement |
| -------- | ----------- | -------- | ----------- |
| Accuracy | 91.55%      | 90.99%   | -0.56%      |
| ECE      | 0.0811      | 0.0267   | **-67%**    |
| NLL      | 1.236       | 0.513    | **-58%**    |

### Noise Robustness (30% label noise)

| Method      | Accuracy Drop |
| ----------- | ------------- |
| Uniform kNN | 0.04%         |
| Attn-KNN    | 1.05%         |

**Finding**: Attention is LESS robust to noise than uniform kNN.

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

### Experiment 5: Enhanced Training with TTA

- **Goal**: Maximize calibration
- **Changes**: TTA, MixUp, label smoothing, optimized config
- **Result**: 78% ECE reduction with TTA
- **Discovery**: TTA is the real innovation, not attention

### Experiment 6: Final Validation

- **Goal**: Reproduce Experiment 5
- **Result**: Pattern confirmed, TTA consistently improves calibration

### Experiment 7: Final Release [CONCLUDED]

- **Goal**: Best possible results with ResNet50
- **Config**: ResNet50, 256-dim, 4-heads, 50 epochs
- **Result**: 91.55% accuracy, 0.0267 ECE (with TTA)
- **Discovery**: Attention is LESS robust to noise than uniform kNN
- **Conclusion**: Project concluded, no further investigation needed

---

## Claim Validation

| Claim                   | Status          | Evidence                                                |
| ----------------------- | --------------- | ------------------------------------------------------- |
| Calibration Improvement | PARTIAL         | Only with TTA (67% ECE reduction)                       |
| Robustness              | **INVALIDATED** | Attention is LESS robust to noise (1.05% drop vs 0.04%) |
| Accuracy Improvement    | **INVALIDATED** | +0.02% is within noise margin                           |
| Compute Overhead        | VALIDATED       | <1ms additional per query                               |

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

**Project Status: CONCLUDED** - No further investigation planned.

### If Others Want to Extend

1. Test on different domains (tabular, NLP)
2. Try transformer embeddings (CLIP, DINO)
3. Focus on TTA optimization rather than attention

### Recommendation

Use distance-weighted kNN with TTA for production. Attention adds complexity without meaningful benefit.

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

- **Experiments**: 7 major iterations
- **Best ECE**: 0.0267 (67% reduction with TTA)
- **Best Accuracy**: 91.55% (Attn-KNN)
- **Gap to CNN**: 4.96%
- **Noise Tolerance**: Uniform kNN better than attention
- **Compute Overhead**: <1ms per query
- **k Robustness**: Stable across k ∈ [1, 50]
- **Status**: CONCLUDED - December 2025

---

## Related Documents

- **research_narrative.md**: Complete chronological narrative of the research journey
- **architecture_diagrams.md**: Comprehensive ASCII art diagrams of architecture and workflows
- **experiments.md**: Detailed experiment documentation
- **honest_assessment.md**: Transparent evaluation of limitations

---

_For complete chronological narrative, see research_narrative.md_
