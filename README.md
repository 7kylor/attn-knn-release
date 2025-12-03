# Attn-KNN: Attention-Weighted K-Nearest Neighbor Classification

## Research Summary

**Core Claim**: Learned attention over neighbors improves calibration and robustness versus uniform and distance-weighted kNN, with minimal compute overhead.

**Status**: CONCLUDED - December 2025. No further investigation needed.

| Claim                   | Status          | Evidence                                                |
| ----------------------- | --------------- | ------------------------------------------------------- |
| Calibration Improvement | PARTIAL         | Only with TTA (67% ECE reduction)                       |
| Robustness              | **INVALIDATED** | Attention is LESS robust to noise (1.05% vs 0.04% drop) |
| Accuracy Improvement    | **INVALIDATED** | +0.02% is within noise margin                           |
| Compute Overhead        | VALIDATED       | <1ms additional per query                               |

---

## Honest Assessment

### What Works

- Test-Time Augmentation (TTA) improves calibration (67% ECE reduction)
- Minimal computational overhead (<1ms per query)
- Distance-weighted kNN is simple and effective

### What Does Not Work

- Attention provides only +0.02% accuracy improvement (negligible)
- Attention is LESS robust to label noise than uniform kNN
- 5% gap to CNN upper bound remains
- Core novelty claim (attention beats baselines) is NOT supported

### Recommendation

**Use distance-weighted kNN with TTA for production.** Attention adds complexity without meaningful benefit. The calibration improvements come from TTA, not from the attention mechanism.

---

## Final Results (Experiment 7 - December 2025)

| Method            | Accuracy   | ECE        | NLL      |
| ----------------- | ---------- | ---------- | -------- |
| Uniform kNN       | 91.53%     | 0.0796     | 1.225    |
| Distance kNN      | 91.52%     | 0.0783     | 1.225    |
| Attn-KNN          | 91.55%     | 0.0811     | 1.236    |
| Attn-KNN + TTA    | 90.99%     | **0.0267** | **0.513**|
| CNN (Upper Bound) | **96.51%** | 0.0253     | 0.184    |

### Noise Robustness (30% label noise)

| Method      | Accuracy Drop |
| ----------- | ------------- |
| Uniform kNN | 0.04%         |
| Attn-KNN    | 1.05%         |

**Finding**: Attention is LESS robust to noise than uniform kNN.

---

## Repository Structure

```
attn-knn-release/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   └── AttnKNN_Experiment.ipynb # Main experiment notebook (self-contained)
│
├── results/                     # Experiment results (versioned by run)
│   ├── experiment_1-3/          # Initial prototype results
│   ├── experiment_4/            # Architecture improvements
│   ├── experiment_5/            # Enhanced training (TTA)
│   ├── experiment_6/            # Final validation
│   ├── summary/                 # Aggregated results and PDFs
│   └── run_YYYYMMDD_HHMMSS/     # New runs saved here with timestamp
│
├── figures/                     # Visualizations
│   ├── reliability_diagrams/    # Calibration plots
│   ├── k_sweep/                 # k parameter analysis
│   ├── robustness/              # Noise/imbalance tests
│   └── comparison/              # Method comparisons
│
├── docs/                        # Documentation
│   ├── method.md                # Method description
│   ├── related_work.md          # Related work
│   ├── future_directions.md     # Future directions
│   ├── experiments.md           # Detailed experiment documentation
│   ├── honest_assessment.md     # Honest limitations and findings
│   ├── research_narrative.md    # Complete chronological research story
│   ├── research_summary.md      # Executive summary with key results
│   └── architecture_diagrams.md # ASCII art architecture diagrams
│
└── data/                        # Datasets (download separately)
    ├── adult/                   # UCI Adult dataset (census income prediction)
    ├── cifar-10-batches-py/     # CIFAR-10 (32x32 color images, 10 classes)
    ├── cifar-100-python/        # CIFAR-100 (32x32 color images, 100 classes)
    ├── ImageNet/                # ImageNet dataset (large-scale image classification)
    ├── iris/                    # Iris flower dataset (classification)
    ├── MNIST/                   # MNIST handwritten digits (28x28 grayscale)
    └── wine-quality/            # Wine quality dataset (regression/classification)
```

---

## Datasets

The repository contains the following datasets in the `data/` folder:

### Image Classification Datasets

- **CIFAR-10** (`cifar-10-batches-py/`)

  - 32×32 color images, 10 classes
  - 50,000 training + 10,000 test images
  - **Status**: Used in experiments (Experiment 1-6)
  - Auto-downloads if not present

- **CIFAR-100** (`cifar-100-python/`)

  - 32×32 color images, 100 classes
  - 50,000 training + 10,000 test images
  - **Status**: Available, not yet used in experiments

- **ImageNet** (`ImageNet/`)

  - Large-scale image classification dataset
  - Organized in train.X1-X4 and val.X subdirectories
  - **Status**: Available, not yet used in experiments

- **MNIST** (`MNIST/`)
  - 28×28 grayscale handwritten digits
  - 60,000 training + 10,000 test images
  - 10 classes (digits 0-9)
  - **Status**: Available, not yet used in experiments

### Tabular/Structured Datasets

- **Adult** (`adult/`)

  - UCI Adult dataset for census income prediction
  - Binary classification (income >50K or ≤50K)
  - Mixed categorical and numerical features
  - **Status**: Available, not yet used in experiments

- **Iris** (`iris/`)

  - Classic flower classification dataset
  - 150 samples, 4 features, 3 classes
  - **Status**: Available, not yet used in experiments

- **Wine Quality** (`wine-quality/`)
  - Wine quality prediction dataset
  - Regression/classification task
  - **Status**: Available, not yet used in experiments

**Note**: The `data/` folder is excluded from git tracking (see `.gitignore`). Datasets should be downloaded separately or will be auto-downloaded by the code when needed.

---

## Quick Start

### Prerequisites

- [UV](https://github.com/astral-sh/uv) (Python package manager)
- Python 3.10+ (UV will manage Python versions automatically)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/attn-knn.git
cd attn-knn

# Create virtual environment and install dependencies
uv venv --python 3.14
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install --upgrade -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=attn-knn --display-name="Python (attn-knn)"
```

### Run Experiment

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/AttnKNN_Experiment.ipynb
# or
jupyter lab notebooks/AttnKNN_Experiment.ipynb
```

**Important**: Select the `Python (attn-knn)` kernel in your notebook (Kernel → Change Kernel).

Run all cells sequentially. The notebook will:

1. Load CIFAR-10 dataset (auto-downloads if needed)
2. Train Attn-KNN model with ResNet50 backbone
3. Train CNN baseline for comparison
4. Evaluate all methods (Uniform, Distance, Attention, TTA)
5. Run k-sweep and noise robustness experiments
6. Generate visualizations and save results

Results are saved to `results/run_YYYYMMDD_HHMMSS/` with timestamp versioning.

---

## Key Findings

### 1. Test-Time Augmentation is the Real Contribution

TTA provides meaningful improvements:

- 67% reduction in ECE (calibration)
- 58% reduction in NLL
- This works with ANY kNN method, not specific to attention

### 2. Attention Does NOT Beat Baselines

- Attention: 91.55% vs Uniform: 91.53% (+0.02%)
- This difference is within noise margin
- Distance-weighted kNN is simpler and equally effective

### 3. Attention is LESS Robust to Noise

- Uniform kNN at 30% noise: 0.04% accuracy drop
- Attn-KNN at 30% noise: 1.05% accuracy drop
- This INVALIDATES the robustness claim

### 4. k Parameter Robustness

- Accuracy stable across k in [1, 50]
- Optimal k is 10-20 for CIFAR-10

---

## Citation

```bibtex
@misc{attnknn2025,
  title={Attn-KNN: Attention-Weighted K-Nearest Neighbor Classification},
  author={[Author Name]},
  year={2025},
  note={Research concluded - attention provides no benefit over distance-weighted kNN. TTA improves calibration.}
}
```

---

## References

- kNN Attention Demystified (arXiv:2411.04013)
- FAISS: Efficient similarity search (Facebook AI)
- Deep Residual Learning (He et al., 2016)

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This research was conducted with honest reporting of both successes and limitations. The project concluded in December 2025 with the finding that attention-weighted kNN does not meaningfully outperform simpler baselines. See `docs/honest_assessment.md` for detailed analysis.

**Project Status**: CONCLUDED - No further investigation planned.
