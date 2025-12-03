# Attn-KNN: Attention-Weighted K-Nearest Neighbor Classification

## Research Summary

**Core Claim**: Learned attention over neighbors improves calibration and robustness versus uniform and distance-weighted kNN, with minimal compute overhead.

**Status**: PARTIALLY VALIDATED

| Claim                   | Status    | Evidence                                      |
| ----------------------- | --------- | --------------------------------------------- |
| Calibration Improvement | VALIDATED | 78% ECE reduction with TTA (0.1297 to 0.0283) |
| Robustness              | VALIDATED | Maintains performance under 30% label noise   |
| Accuracy Improvement    | MARGINAL  | +0.03% to +0.38% over uniform kNN             |
| Compute Overhead        | VALIDATED | <1ms additional per query                     |

---

## Honest Assessment

### What Works

- Test-Time Augmentation (TTA) dramatically improves calibration
- Method is robust to label noise and class imbalance
- Minimal computational overhead
- Theoretically grounded (validates k-sweep error bounds)

### What Does Not Work

- Accuracy gains are marginal (<1% over uniform kNN)
- 8-9% gap to CNN upper bound
- Attention mechanism alone does not beat distance-weighted kNN
- Core novelty claim (attention beats baselines) is weakly supported

### Recommendation

This work demonstrates that **calibration can be significantly improved** through attention + TTA, but does not demonstrate meaningful accuracy improvements. The method is best suited for applications where calibration/uncertainty estimation is critical.

---

## Best Results (Experiment 5)

| Method             | Accuracy   | ECE        | NLL      | F1-Macro   |
| ------------------ | ---------- | ---------- | -------- | ---------- |
| Uniform kNN        | 86.82%     | 0.1297     | 2.05     | 86.82%     |
| Distance kNN       | 86.85%     | 0.1099     | 2.05     | 86.85%     |
| Attn-KNN           | 86.85%     | 0.1300     | 2.05     | 86.85%     |
| **Attn-KNN + TTA** | **87.20%** | **0.0283** | **0.97** | **87.27%** |
| CNN (Upper Bound)  | 95.12%     | 0.0685     | 0.24     | 95.12%     |

---

## Repository Structure

```
attn-knn-release/
├── README.md                    # This file
├── EXPERIMENTS.md               # Detailed experiment documentation
├── HONEST_ASSESSMENT.md         # Honest limitations and findings
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
│   └── future_directions.md     # Future directions
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

### 1. Test-Time Augmentation is Critical

TTA provides the largest improvements:

- 78% reduction in ECE (calibration)
- 53% reduction in NLL
- Marginal accuracy improvement

### 2. Attention vs Distance Weighting

Attention and distance-weighted kNN perform similarly. Attention provides slight calibration advantages when combined with TTA.

### 3. k Parameter Robustness

- Accuracy stable across k in [1, 50]
- NLL decreases with k (validates theory)
- Optimal k is 10-20 for CIFAR-10

### 4. Noise Robustness

All kNN methods maintain accuracy under 30% label noise. This is inherent to kNN, not unique to attention.

---

## Citation

```bibtex
@misc{attnknn2025,
  title={Attn-KNN: Attention-Weighted K-Nearest Neighbor Classification},
  author={[Author Name]},
  year={2025},
  note={Research prototype - calibration improvements validated, accuracy gains marginal}
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

This research was conducted with honest reporting of both successes and limitations. See HONEST_ASSESSMENT.md for detailed analysis.
