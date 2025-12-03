# Related Work

## kNN in Deep Learning

### kNN-LM (Khandelwal et al., 2020)

**"Generalization through Memorization: Nearest Neighbor Language Models"**

- Interpolates language model predictions with kNN over datastore
- Improves perplexity without retraining
- Uses fixed interpolation weight

**Relation to Our Work**: I learn attention weights instead of using fixed interpolation.

### RETRO (Borgeaud et al., 2022)

**"Improving Language Models by Retrieving from Trillions of Tokens"**

- Retrieval-enhanced transformer at scale
- Cross-attention over retrieved chunks
- Trained end-to-end

**Relation to Our Work**: Similar attention-over-neighbors idea, but for language modeling.

### kNN Attention (arXiv:2411.04013)

**"kNN Attention Demystified"**

- Theoretical analysis of kNN attention
- Error bounds for kNN classification
- Connection between k and error rate

**Relation to Our Work**: Direct theoretical foundation for our approach.

---

## Metric Learning

### ProxyAnchor (Kim et al., 2020)

**"Proxy Anchor Loss for Deep Metric Learning"**

- Learns class proxies for efficient training
- Better than triplet loss for large classes
- Stable optimization

**Relation to Our Work**: Alternative to contrastive loss for embedding learning.

### Supervised Contrastive Learning (Khosla et al., 2020)

**"Supervised Contrastive Learning"**

- Extends SimCLR to supervised setting
- Pulls together same-class samples
- Pushes apart different-class samples

**Relation to Our Work**: I use this for embedding training (with margin).

---

## Calibration

### Temperature Scaling (Guo et al., 2017)

**"On Calibration of Modern Neural Networks"**

- Post-hoc calibration method
- Single learned temperature parameter
- Achieves good ECE with minimal changes

**Relation to Our Work**: I use per-head learned temperature; baseline comparison.

### Focal Loss (Lin et al., 2017)

**"Focal Loss for Dense Object Detection"**

- Down-weights easy examples
- Implicit calibration effect
- Designed for class imbalance

**Relation to Our Work**: Alternative loss function I did not explore.

---

## Attention Mechanisms

### Multi-Head Attention (Vaswani et al., 2017)

**"Attention Is All You Need"**

- Parallel attention heads
- Query-key-value formulation
- Foundation for transformers

**Relation to Our Work**: I adapt multi-head attention for neighbor weighting.

### Self-Attention in Vision (Dosovitskiy et al., 2020)

**"An Image is Worth 16x16 Words"**

- Vision Transformer (ViT)
- Self-attention over image patches
- State-of-the-art image classification

**Relation to Our Work**: Alternative embedding model I could explore.

---

## Non-Parametric Methods

### Prototypical Networks (Snell et al., 2017)

**"Prototypical Networks for Few-shot Learning"**

- Class prototypes from embedding means
- Distance-based classification
- Effective for few-shot learning

**Relation to Our Work**: Similar non-parametric approach; prototypes vs kNN.

### Matching Networks (Vinyals et al., 2016)

**"Matching Networks for One Shot Learning"**

- Attention over support set
- Full context embedding
- End-to-end training

**Relation to Our Work**: Most similar prior work; I focus on calibration.

---

## Test-Time Augmentation

### TTA for Uncertainty (Ayhan & Berens, 2018)

**"Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty"**

- Multiple augmented views at test time
- Variance as uncertainty estimate
- Improves calibration

**Relation to Our Work**: TTA is our key finding for calibration improvement.

---

## Key Differences from Prior Work

| Aspect      | Prior Work                   | Our Work                            |
| ----------- | ---------------------------- | ----------------------------------- |
| Attention   | Fixed weights or single-head | Multi-head with learned temperature |
| Training    | Often frozen embeddings      | End-to-end with kNN loss            |
| Calibration | Post-hoc only                | In-training + post-hoc              |
| TTA         | Rarely explored              | Key contribution                    |
| Focus       | Accuracy                     | Accuracy + Calibration              |

---

## What I Learned from Related Work

1. **Temperature matters**: Both for attention (learned) and calibration (scaling)
2. **kNN is powerful**: Can match sophisticated models with good embeddings
3. **Calibration is hard**: Requires specific techniques (TTA, temperature scaling)
4. **Scale matters**: Most prior work on larger datasets shows clearer benefits

---

## Gaps in Related Work

1. **Calibration in kNN**: Rarely studied explicitly
2. **Attention for kNN classification**: Limited work on learned neighbor weights
3. **TTA for non-parametric methods**: Underexplored

These gaps motivated our research direction.
