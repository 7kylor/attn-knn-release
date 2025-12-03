# Future Directions

## High Priority

### 1. Larger Scale Evaluation

**Current Limitation**: CIFAR-10 may be saturated; results don't generalize.

**Proposed Work**:

- ImageNet-100 or ImageNet-1K subset
- Fine-grained classification (CUB-200, Stanford Cars)
- Domains where kNN has advantages (few-shot, retrieval)

**Expected Outcome**: Better understanding of when attention helps.

### 2. Transformer Embeddings

**Current Limitation**: ResNet embeddings may not benefit from attention.

**Proposed Work**:

- CLIP embeddings (vision-language)
- DINO embeddings (self-supervised)
- ViT embeddings (attention-based)

**Hypothesis**: Attention over transformer embeddings may be more effective than attention over CNN embeddings.

### 3. Calibration-First Objective

**Current Limitation**: Training optimizes accuracy, calibration is byproduct.

**Proposed Work**:

- Add differentiable ECE loss during training
- Temperature learning with calibration target
- Selective prediction training objective

**Expected Outcome**: Better calibration without relying on TTA.

---

## Medium Priority

### 4. Adaptive k Selection

**Current Limitation**: Fixed k for all queries.

**Proposed Work**:

- Learn to predict optimal k per query
- Entropy-based stopping criterion
- Confidence-based k selection

**Hypothesis**: Different queries benefit from different k values.

### 5. Hierarchical Attention

**Current Limitation**: Flat attention over all neighbors.

**Proposed Work**:

- Cluster neighbors first
- Attention over clusters, then within clusters
- Multi-scale neighbor aggregation

**Expected Outcome**: Better handling of diverse neighbor sets.

### 6. Online Learning

**Current Limitation**: Memory bank is static after training.

**Proposed Work**:

- Continual memory bank updates
- Importance-weighted memory
- Forgetting mechanisms for old samples

**Expected Outcome**: Adaptation to distribution shift.

---

## Lower Priority

### 7. Cross-Modal Retrieval

**Application**: Image-to-text, text-to-image retrieval.

**Proposed Work**:

- CLIP-based embeddings
- Cross-modal attention
- Calibrated retrieval confidence

**Connection**: Relates to RAG uncertainty direction.

### 8. Explainability

**Current Limitation**: Attention weights are not interpretable.

**Proposed Work**:

- Attention visualization
- Neighbor attribution
- Counterfactual explanations

**Expected Outcome**: Understanding of what attention learns.

### 9. Efficiency Improvements

**Current Limitation**: Memory bank search is O(n).

**Proposed Work**:

- Learned hash functions
- Hierarchical indices
- Pruning strategies

**Expected Outcome**: Scalability to millions of samples.

---

## Not Recommended

### A. More Complex Attention

**Why Not**: Current results show multi-head doesn't beat single-head. Adding complexity unlikely to help.

### B. Larger Models

**Why Not**: The bottleneck is the kNN paradigm, not model capacity. CNN upper bound is ~95% vs ~87% for kNN.

### C. More Training

**Why Not**: Experiments 4-6 show diminishing returns beyond 20 epochs.

---

## Pivot Directions

If attention-weighted kNN continues to show marginal benefits, consider:

### Direction A: Calibrated Retrieval Confidence

**Focus**: Apply calibration techniques to RAG systems.
**Rationale**: Leverages calibration expertise in high-impact domain.
**See**: `docs/CALIBRATED_RETRIEVAL_CONFIDENCE_PLAN.md`

### Direction B: Uncertainty Quantification for kNN

**Focus**: When should kNN abstain?
**Rationale**: Calibration results suggest kNN can be well-calibrated.
**Approach**: Focus on selective prediction, not accuracy.

### Direction C: kNN for Few-Shot Learning

**Focus**: Domains with limited labeled data.
**Rationale**: kNN naturally handles few-shot scenarios.
**Approach**: Meta-learning for attention weights.

---

## Research Questions

Based on this project's findings, open questions include:

1. **Why doesn't attention beat distance weighting?**

   - Is the learned temperature equivalent to fixed temperature?
   - Is attention collapsing to uniform?

2. **Why does TTA help so much?**

   - Is it reducing embedding variance?
   - Is it providing implicit ensemble?

3. **What is the true ceiling for kNN?**

   - Can embeddings be improved to close CNN gap?
   - Is 8-9% gap fundamental?

4. **When does attention help in kNN?**
   - Different data distributions?
   - Different embedding models?
   - Different neighbor characteristics?

---

## Concrete Next Steps

If continuing this project:

1. **Week 1-2**: Run ImageNet-100 experiments
2. **Week 3-4**: Try CLIP/DINO embeddings
3. **Week 5-6**: Implement differentiable ECE loss
4. **Week 7-8**: Analyze results, decide on pivot

If pivoting:

- See `docs/CALIBRATED_RETRIEVAL_CONFIDENCE_PLAN.md` for RAG uncertainty direction
- Consider few-shot learning application
- Focus on calibration as primary contribution
