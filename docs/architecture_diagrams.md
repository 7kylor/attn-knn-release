# Attn-KNN Architecture Diagrams and Illustrations

This document contains comprehensive ASCII art diagrams illustrating the architecture, novel components, and workflows of the Attn-KNN system.

---

## Table of Contents

- [Attn-KNN Architecture Diagrams and Illustrations](#attn-knn-architecture-diagrams-and-illustrations)
  - [Table of Contents](#table-of-contents)
  - [1. Overall System Architecture](#1-overall-system-architecture)
  - [2. Embedding Network Architecture](#2-embedding-network-architecture)
  - [3. Multi-Head Neighbor Attention Mechanism](#3-multi-head-neighbor-attention-mechanism)
  - [4. Novel Components](#4-novel-components)
    - [4.1 Neighbor Self-Attention](#41-neighbor-self-attention)
    - [4.2 Label-Conditioned Attention Bias](#42-label-conditioned-attention-bias)
    - [4.3 Prototype-Guided Scoring](#43-prototype-guided-scoring)
  - [5. Training Workflow](#5-training-workflow)
  - [6. Inference Workflow](#6-inference-workflow)
  - [7. Test-Time Augmentation (TTA)](#7-test-time-augmentation-tta)
  - [8. Comparison with Baselines](#8-comparison-with-baselines)
  - [9. Memory Bank Architecture](#9-memory-bank-architecture)
  - [10. Loss Function Components](#10-loss-function-components)
  - [Summary of Novel Components](#summary-of-novel-components)

---

## 1. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Attn-KNN System Architecture                      │
└─────────────────────────────────────────────────────────────────────────┘

Input Image (32×32×3)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING NETWORK                                  │
│                                                                           │
│  ResNet50 Backbone (ImageNet Pretrained)                                 │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ Conv1 (3→64, 3×3) → BN → ReLU                      │                │
│  │ Layer1 (64→256) × 3 blocks                          │                │
│  │ Layer2 (256→512) × 4 blocks                         │                │
│  │ Layer3 (512→1024) × 6 blocks                        │                │
│  │ Layer4 (1024→2048) × 3 blocks                        │                │
│  │ Global Average Pooling                               │                │
│  └─────────────────────────────────────────────────────┘                │
│                          │                                                │
│                          ▼                                                │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ PROJECTION HEAD                                      │                │
│  │ Linear(2048 → 1024) → LayerNorm → GELU → Dropout(0.1)│                │
│  │ Linear(1024 → 512) → LayerNorm → GELU → Dropout(0.1)│                │
│  │ Linear(512 → 256)                                    │                │
│  │ L2 Normalization                                     │                │
│  └─────────────────────────────────────────────────────┘                │
│                          │                                                │
│                          ▼                                                │
│              Query Embedding (256-dim, L2-normalized)                     │
└───────────────────────────────────────────────────────────────────────────┘
        │
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         MEMORY BANK (FAISS)                               │
│                                                                           │
│  Training Set Embeddings (50,000 samples)                                │
│  ┌─────────────────────────────────────────────────────┐                │
│  │ FAISS Index (Flat L2 or HNSW)                        │                │
│  │ • Fast k-NN search                                    │                │
│  │ • GPU-accelerated                                     │                │
│  │ • <0.05ms per query                                   │                │
│  └─────────────────────────────────────────────────────┘                │
│                          │                                                │
│                          ▼                                                │
│              Top-k Nearest Neighbors (k=16)                               │
│              • Neighbor Embeddings (B, K, 256)                            │
│              • Neighbor Labels (B, K)                                    │
│              • Distances (B, K)                                          │
└───────────────────────────────────────────────────────────────────────────┘
        │
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              MULTI-HEAD NEIGHBOR ATTENTION (MHNA)                        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ NOVEL: Neighbor Self-Attention (Optional)                        │   │
│  │ Neighbors attend to each other to refine representations         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                                │
│                          ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Query-Key Attention                                               │   │
│  │ Q = Query @ W_q  (B, 1, H, d)                                    │   │
│  │ K = Neighbors @ W_k  (B, K, H, d)                                │   │
│  │ Scores = (Q @ K^T) / (√d × τ)  (B, H, 1, K)                      │   │
│  │                                                                   │   │
│  │ Learned Temperature: τ ∈ [0.05, 2.0] per head                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                                │
│                          ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Attention Score Modifications                                    │   │
│  │                                                                   │   │
│  │ 1. Distance Bias Network                                          │   │
│  │    dists → MLP(64) → bias per head                               │   │
│  │                                                                   │   │
│  │ 2. NOVEL: Label-Conditioned Bias                                  │   │
│  │    neighbor_labels → one-hot → mean → MLP(64) → bias             │   │
│  │                                                                   │   │
│  │ 3. NOVEL: Prototype-Guided Scoring                                │   │
│  │    neighbors ↔ class_prototypes → alignment → bias                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                                │
│                          ▼                                                │
│  Attention Weights: α = softmax(scores + biases)  (B, K)                │
└───────────────────────────────────────────────────────────────────────────┘
        │
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    WEIGHTED CLASSIFICATION                                │
│                                                                           │
│  Neighbor Labels (B, K) → One-Hot (B, K, C)                              │
│                          │                                                │
│                          ▼                                                │
│  Class Probabilities = α @ One-Hot(neighbor_labels)  (B, C)              │
│                          │                                                │
│                          ▼                                                │
│              Prediction = argmax(probabilities)                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Embedding Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ImageEmbedder Architecture                            │
└─────────────────────────────────────────────────────────────────────────┘

Input: Image (B, 3, 32, 32)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ ResNet50 Backbone (ImageNet Pretrained, Fully Unfrozen)                 │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Modified Conv1 Layer                                          │     │
│  │ • Original: 7×7 conv, stride=2, padding=3                     │     │
│  │ • Modified: 3×3 conv, stride=1, padding=1 (for 32×32 images) │     │
│  │ • Output: (B, 64, 32, 32)                                      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ ResNet Blocks                                                  │     │
│  │                                                                 │     │
│  │ Layer1: 64 → 256 channels, ×3 blocks                           │     │
│  │   ┌─────────────────────────────────────────────┐             │     │
│  │   │ Bottleneck: 1×1 → 3×3 → 1×1                │             │     │
│  │   │ Residual connection                          │             │     │
│  │   └─────────────────────────────────────────────┘             │     │
│  │                                                                 │     │
│  │ Layer2: 256 → 512 channels, ×4 blocks                          │     │
│  │ Layer3: 512 → 1024 channels, ×6 blocks                        │     │
│  │ Layer4: 1024 → 2048 channels, ×3 blocks                        │     │
│  │                                                                 │     │
│  │ Output: (B, 2048, 1, 1)                                        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Global Average Pooling                                         │     │
│  │ (B, 2048, 1, 1) → (B, 2048)                                   │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
┌───────────────────────────────────────────────────────────────────────────┐
│ Projection Head                                                           │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Linear(2048 → 1024)                                            │     │
│  │ LayerNorm(1024)                                                 │     │
│  │ GELU                                                            │     │
│  │ Dropout(0.1)                                                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Linear(1024 → 512)                                              │     │
│  │ LayerNorm(512)                                                  │     │
│  │ GELU                                                            │     │
│  │ Dropout(0.1)                                                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Linear(512 → 256)                                               │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ L2 Normalization                                                │     │
│  │ e = e / ||e||₂                                                  │     │
│  │ Ensures embeddings lie on unit sphere                           │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│              Output: Embedding (B, 256), ||e||₂ = 1                      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Head Neighbor Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Multi-Head Neighbor Attention (MHNA) Detailed Flow            │
└─────────────────────────────────────────────────────────────────────────┘

Inputs:
  • Query Embedding: q (B, 256)
  • Neighbor Embeddings: N (B, K, 256), K=16 neighbors
  • Distances: d (B, K)
  • Neighbor Labels: y_n (B, K)

        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Neighbor Self-Attention (NOVEL)                                 │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ NeighborSelfAttention Module                                   │     │
│  │                                                                 │     │
│  │ N (B, K, 256)                                                  │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ QKV = N @ W_qkv  → (B, K, 768)                                │     │
│  │ Split → Q, K, V  (each: B, K, 256)                            │     │
│  │                                                                 │     │
│  │ Reshape for Multi-Head:                                        │     │
│  │ Q, K, V → (B, H, K, d) where H=4 heads, d=64                  │     │
│  │                                                                 │     │
│  │ Attention:                                                     │     │
│  │ scores = (Q @ K^T) / √d  (B, H, K, K)                         │     │
│  │ attn = softmax(scores)                                        │     │
│  │ out = attn @ V  (B, H, K, d)                                  │     │
│  │                                                                 │     │
│  │ Reshape → (B, K, 256)                                          │     │
│  │ Output Projection: out @ W_out                                 │     │
│  │ Residual: N + out                                              │     │
│  │ LayerNorm                                                       │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│              Refined Neighbors: N' (B, K, 256)                           │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Query-Key Projection                                             │
│                                                                           │
│  Query: q (B, 256)                                                       │
│    │                                                                      │
│    ▼                                                                      │
│  Q = q @ W_q  → (B, 256)                                                 │
│    Reshape → (B, 1, H, d) where H=4, d=64                               │
│    Transpose → (B, H, 1, d)                                              │
│                                                                           │
│  Neighbors: N' (B, K, 256)                                               │
│    │                                                                      │
│    ▼                                                                      │
│  K = N' @ W_k  → (B, K, 256)                                            │
│    Reshape → (B, K, H, d)                                                 │
│    Transpose → (B, H, K, d)                                               │
│                                                                           │
│  V = N' @ W_v  → (B, K, 256)                                            │
│    Reshape → (B, K, H, d)                                                 │
│    Transpose → (B, H, K, d)                                               │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Attention Score Computation                                      │
│                                                                           │
│  Base Score:                                                              │
│  scores_base = Q @ K^T  (B, H, 1, K)                                     │
│  scores_base = scores_base / (√d × τ)                                    │
│                                                                           │
│  where:                                                                   │
│    • d = 64 (head dimension)                                             │
│    • τ = learned temperature per head, clamped [0.05, 2.0]               │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Temperature Learning                                             │     │
│  │                                                                 │     │
│  │ log_τ (H,) ← learned parameter                                 │     │
│  │ τ = exp(log_τ).clamp(0.05, 2.0)                                │     │
│  │                                                                 │     │
│  │ Each head learns its own temperature                            │     │
│  │ Initial: τ₀ = 0.5                                              │     │
│  └───────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Attention Score Modifications (Novel Components)                │
│                                                                           │
│  scores = scores_base                                                     │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Modification 1: Distance Bias                                  │     │
│  │                                                                 │     │
│  │ d (B, K) → unsqueeze(-1) → (B, K, 1)                          │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ MLP: Linear(1→64) → GELU → Linear(64→64) → GELU → Linear(64→H)│     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ dist_bias (B, K, H)                                            │     │
│  │ Permute → (B, H, K)                                            │     │
│  │ Unsqueeze → (B, H, 1, K)                                       │     │
│  │                                                                 │     │
│  │ scores += dist_bias                                             │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Modification 2: Label-Conditioned Bias (NOVEL)               │     │
│  │                                                                 │     │
│  │ y_n (B, K) → one_hot → (B, K, C) where C=10 classes           │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ Mean over K → label_dist (B, C)                               │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ MLP: Linear(C→64) → GELU → Linear(64→H)                       │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ label_bias (B, H)                                             │     │
│  │ Unsqueeze → (B, H, 1, 1)                                       │     │
│  │                                                                 │     │
│  │ scores += label_bias  (broadcasted)                            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Modification 3: Prototype-Guided Scoring (NOVEL)             │     │
│  │                                                                 │     │
│  │ Class Prototypes: P (C, 256) ← learned parameters            │     │
│  │   │                                                            │     │
│  │   ▼                                                            │     │
│  │ P_norm = L2_normalize(P)                                      │     │
│  │ N_norm = L2_normalize(N')                                      │     │
│  │                                                                 │     │
│  │ For each neighbor:                                             │     │
│  │   proto_idx = y_n[i]  (class index)                            │     │
│  │   proto_i = P_norm[proto_idx]  (B, K, 256)                    │     │
│  │                                                                 │     │
│  │ Alignment:                                                     │     │
│  │   align = (N_norm * proto_i).sum(dim=-1)  (B, K)              │     │
│  │   proto_bias = align × scale  (B, K)                          │     │
│  │   Unsqueeze → (B, 1, 1, K)                                     │     │
│  │                                                                 │     │
│  │ scores += proto_bias                                           │     │
│  └───────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Attention Weight Computation                                     │
│                                                                           │
│  attn_weights = softmax(scores, dim=-1)  (B, H, 1, K)                   │
│                                                                           │
│  Average over heads:                                                      │
│  attn = attn_weights.squeeze(2).mean(dim=1)  (B, K)                     │
│                                                                           │
│  Output: Attention weights α (B, K), Σα = 1                              │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Novel Components

### 4.1 Neighbor Self-Attention

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Neighbor Self-Attention (NOVEL Component)                  │
└─────────────────────────────────────────────────────────────────────────┘

Motivation:
  Neighbors can attend to each other to refine their representations
  based on local neighborhood structure before query aggregation.

Architecture:

  Neighbors N (B, K, 256)
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ QKV Projection                                                │
  │ N @ W_qkv → (B, K, 768)                                       │
  │ Split → Q, K, V (each: B, K, 256)                            │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Multi-Head Reshape                                           │
  │ Q, K, V → (B, H, K, d) where H=4, d=64                       │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Self-Attention                                                │
  │                                                               │
  │     N₁ ──┐                                                    │
  │     N₂ ──┤                                                    │
  │     N₃ ──┼──→ Attention Matrix (K×K)                        │
  │     ...  │                                                    │
  │     Nₖ ──┘                                                    │
  │                                                               │
  │  Each neighbor attends to all neighbors                       │
  │  scores = (Q @ K^T) / √d                                     │
  │  attn = softmax(scores)                                      │
  │  out = attn @ V                                              │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Output Projection & Residual                                 │
  │ out @ W_out → (B, K, 256)                                     │
  │ N' = LayerNorm(N + out)                                       │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  Refined Neighbors N' (B, K, 256)

Visual Representation:

  Before Self-Attention:          After Self-Attention:

    N₁ ──┐                          N₁' ──┐
    N₂ ──┤                          N₂' ──┤
    N₃ ──┤  (isolated)              N₃' ──┤  (context-aware)
    N₄ ──┤                          N₄' ──┤
    N₅ ──┘                          N₅' ──┘

    Each neighbor                   Each neighbor refined
    independent                     based on neighborhood
```

### 4.2 Label-Conditioned Attention Bias

```
┌─────────────────────────────────────────────────────────────────────────┐
│         Label-Conditioned Attention Bias (NOVEL Component)               │
└─────────────────────────────────────────────────────────────────────────┘

Motivation:
  Learn how the distribution of labels among neighbors affects
  attention weights. Different label distributions may require
  different attention patterns.

Architecture:

  Neighbor Labels y_n (B, K)
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ One-Hot Encoding                                              │
  │ y_n → one_hot → (B, K, C) where C=10 classes                 │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Compute Label Distribution                                    │
  │ mean over K dimension → label_dist (B, C)                     │
  │                                                               │
  │ Example:                                                      │
  │   Neighbors: [cat, cat, dog, bird, cat]                       │
  │   Distribution: [0.6, 0.2, 0.2, 0, 0, ...]                  │
  │   (60% cat, 20% dog, 20% bird)                              │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Label Bias Network                                            │
  │                                                               │
  │ label_dist (B, C)                                             │
  │   │                                                           │
  │   ▼                                                           │
  │ Linear(C → 64)                                                │
  │   │                                                           │
  │   ▼                                                           │
  │ GELU                                                          │
  │   │                                                           │
  │   ▼                                                           │
  │ Linear(64 → H) where H=4 heads                                │
  │   │                                                           │
  │   ▼                                                           │
  │ label_bias (B, H)                                             │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Add to Attention Scores                                       │
  │                                                               │
  │ label_bias (B, H)                                             │
  │   Unsqueeze → (B, H, 1, 1)                                    │
  │   Broadcast → (B, H, 1, K)                                   │
  │                                                               │
  │ scores += label_bias                                          │
  └───────────────────────────────────────────────────────────────┘

Example:

  Scenario 1: Uniform label distribution
    Neighbors: [cat, dog, bird, plane, car, ...]
    → Low bias (balanced distribution)

  Scenario 2: Skewed label distribution
    Neighbors: [cat, cat, cat, cat, cat, ...]
    → High bias (learns to adjust attention)
```

### 4.3 Prototype-Guided Scoring

```
┌─────────────────────────────────────────────────────────────────────────┐
│            Prototype-Guided Scoring (NOVEL Component)                   │
└─────────────────────────────────────────────────────────────────────────┘

Motivation:
  Learn class prototypes (centroids) and use alignment between
  neighbors and their class prototypes to guide attention.

Architecture:

  ┌───────────────────────────────────────────────────────────────┐
  │ Learnable Class Prototypes                                    │
  │                                                               │
  │ P (C, 256) ← learned parameters                              │
  │   P[0] = prototype for class 0 (e.g., "airplane")            │
  │   P[1] = prototype for class 1 (e.g., "automobile")         │
  │   ...                                                         │
  │   P[9] = prototype for class 9 (e.g., "truck")              │
  │                                                               │
  │ Initialization: N(0, 0.02²)                                  │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Normalize Prototypes                                          │
  │ P_norm = L2_normalize(P, dim=1)  (C, 256)                    │
  │                                                               │
  │ Ensures prototypes lie on unit sphere                         │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ For Each Neighbor:                                            │
  │                                                               │
  │ Neighbor Embedding: n_i (256)                                │
  │ Neighbor Label: y_i (class index)                            │
  │                                                               │
  │ Get Prototype:                                                │
  │   proto_i = P_norm[y_i]  (256)                               │
  │                                                               │
  │ Normalize Neighbor:                                           │
  │   n_i_norm = L2_normalize(n_i)                               │
  │                                                               │
  │ Compute Alignment:                                            │
  │   align_i = (n_i_norm · proto_i)  (scalar)                   │
  │   = cosine_similarity(n_i, proto_i)                           │
  │                                                               │
  │ High alignment → neighbor matches its class prototype        │
  │ Low alignment → neighbor is atypical for its class            │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Prototype Bias                                                │
  │                                                               │
  │ For all neighbors:                                            │
  │   align (B, K) ← alignment scores                            │
  │   proto_bias = align × scale  (B, K)                         │
  │   where scale ← learned parameter                            │
  │                                                               │
  │ Reshape: (B, K) → (B, 1, 1, K)                               │
  │                                                               │
  │ scores += proto_bias                                           │
  └───────────────────────────────────────────────────────────────┘

Visual Representation:

  Class Prototypes (learned):

    P₀ (airplane) ●
    P₁ (automobile) ●
    P₂ (bird) ●
    ...
    P₉ (truck) ●

  Neighbor Embeddings:

    n₁ (airplane) ●───→ High alignment with P₀ → High bias
    n₂ (airplane) ●───→ Medium alignment with P₀ → Medium bias
    n₃ (bird) ●────────→ High alignment with P₂ → High bias
    n₄ (airplane?) ●──→ Low alignment with P₀ → Low bias
                      (atypical airplane)
```

---

## 5. Training Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Training Workflow                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ INITIALIZATION PHASE                                                      │
│                                                                           │
│ 1. Load Dataset (CIFAR-10)                                              │
│    50,000 training samples, 10,000 test samples                          │
│                                                                           │
│ 2. Initialize Model                                                      │
│    • ImageEmbedder (ResNet50, ImageNet pretrained)                       │
│    • MultiHeadNeighborAttention                                          │
│    • All parameters trainable                                           │
│                                                                           │
│ 3. Build Initial Memory Bank                                             │
│    • Forward pass through training set                                  │
│    • Extract embeddings                                                  │
│    • Build FAISS index                                                   │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (20 epochs)                                                │
│                                                                           │
│ For each epoch:                                                          │
│                                                                           │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ For each batch (B=512):                                        │     │
│   │                                                                 │     │
│   │   1. Load batch: x (B, 3, 32, 32), y (B,)                     │     │
│   │                                                                 │     │
│   │   2. Data Augmentation:                                         │     │
│   │      • RandomHorizontalFlip(p=0.5)                             │     │
│   │      • RandomCrop(32, padding=4)                               │     │
│   │      • MixUp (α=0.4) - NOVEL                                   │     │
│   │      • Label Smoothing (ε=0.1)                                 │     │
│   │                                                                 │     │
│   │   3. Forward Pass:                                              │     │
│   │      query_emb = embedder(x)  (B, 256)                          │     │
│   │                                                                 │     │
│   │   4. Neighbor Retrieval:                                       │     │
│   │      • Search memory bank for k=16 neighbors                  │     │
│   │      • Get neighbor embeddings, labels, distances              │     │
│   │                                                                 │     │
│   │   5. Attention Computation:                                    │     │
│   │      attn, _ = attention(query_emb, neigh_emb,                 │     │
│   │                           dists, neigh_labels)                 │     │
│   │                                                                 │     │
│   │   6. Prediction:                                                │     │
│   │      probs = attn @ one_hot(neigh_labels)  (B, C)             │     │
│   │                                                                 │     │
│   │   7. Loss Computation:                                          │     │
│   │      ┌───────────────────────────────────────────────┐         │     │
│   │      │ Loss Components:                             │         │     │
│   │      │                                               │         │     │
│   │      │ L_knn = CrossEntropy(log(probs), y)         │         │     │
│   │      │                                               │         │     │
│   │      │ L_contrastive = SupervisedContrastive(        │         │     │
│   │      │   query_emb, y, margin=0.5)                  │         │     │
│   │      │                                               │         │     │
│   │      │ L_entropy = -entropy(attn)  (regularization) │         │     │
│   │      │                                               │         │     │
│   │      │ L_total = L_knn + 0.5×L_contrastive          │         │     │
│   │      │            + 0.01×L_entropy                   │         │     │
│   │      └───────────────────────────────────────────────┘         │     │
│   │                                                                 │     │
│   │   8. Backward Pass:                                            │     │
│   │      L_total.backward()                                       │     │
│   │      Gradient clipping (max_norm=1.0)                          │     │
│   │      optimizer.step()                                        │     │
│   │                                                                 │     │
│   │   9. Memory Bank Update (every N batches):                    │     │
│   │      • Recompute embeddings for training set                 │     │
│   │      • Rebuild FAISS index                                    │     │
│   │                                                                 │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                           │
│   End of epoch:                                                           │
│   • Evaluate on validation set                                          │
│   • Save best model (lowest validation loss)                            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ POST-TRAINING                                                             │
│                                                                           │
│ 1. Load best model checkpoint                                            │
│                                                                           │
│ 2. Rebuild final memory bank with best model                             │
│                                                                           │
│ 3. Evaluate on test set                                                  │
│    • Uniform kNN                                                         │
│    • Distance-weighted kNN                                               │
│    • Attention-weighted kNN                                             │
│    • Attention-weighted kNN + TTA                                        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Inference Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Inference Workflow                                   │
└─────────────────────────────────────────────────────────────────────────┘

Test Sample: x_test (3, 32, 32)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Embedding Extraction                                             │
│                                                                           │
│  query_emb = model.get_embedding(x_test)  (256,)                        │
│                                                                           │
│  • Forward pass through ResNet50 + projection head                       │
│  • L2 normalization                                                       │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Neighbor Retrieval (FAISS)                                        │
│                                                                           │
│  dists, indices = memory_bank.search(query_emb, k=16)                    │
│                                                                           │
│  Returns:                                                                 │
│    • dists: distances to k neighbors (16,)                               │
│    • indices: indices of neighbors in memory bank (16,)                  │
│                                                                           │
│  Extract:                                                                 │
│    • neigh_emb = memory_bank.embeddings[indices]  (16, 256)              │
│    • neigh_labels = memory_bank.labels[indices]  (16,)                    │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Attention Computation                                             │
│                                                                           │
│  attn, _ = model.attention(                                              │
│      query_emb.unsqueeze(0),      # (1, 256)                              │
│      neigh_emb.unsqueeze(0),      # (1, 16, 256)                         │
│      dists.unsqueeze(0),          # (1, 16)                               │
│      neigh_labels.unsqueeze(0)     # (1, 16)                               │
│  )                                                                        │
│                                                                           │
│  Returns:                                                                 │
│    • attn: attention weights (1, 16)                                    │
│                                                                           │
│  Process:                                                                 │
│    1. Neighbor self-attention (refine neighbors)                        │
│    2. Query-key attention                                                │
│    3. Apply biases (distance, label, prototype)                           │
│    4. Softmax → attention weights                                        │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Weighted Classification                                          │
│                                                                           │
│  neigh_onehot = one_hot(neigh_labels, num_classes=10)  (16, 10)          │
│                                                                           │
│  probs = attn @ neigh_onehot  (1, 10)                                    │
│                                                                           │
│  Normalize:                                                               │
│    probs = probs / probs.sum()                                           │
│                                                                           │
│  Prediction:                                                              │
│    pred_class = argmax(probs)                                            │
│    confidence = max(probs)                                                │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Test-Time Augmentation (TTA)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Test-Time Augmentation (TTA) Workflow                       │
└─────────────────────────────────────────────────────────────────────────┘

KEY INNOVATION: TTA provides 78% ECE reduction

Original Test Sample: x (3, 32, 32)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Generate Augmented Views                                                 │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ View 0: Original                                             │       │
│  │   x₀ = x                                                     │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ View 1: Horizontal Flip                                     │       │
│  │   x₁ = HorizontalFlip(x)                                     │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ View 2: Random Crop 1                                        │       │
│  │   x₂ = RandomCrop(x, padding=4)                             │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ View 3: Random Crop 2                                        │       │
│  │   x₃ = RandomCrop(x, padding=4)                             │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ View 4: Random Crop 3                                        │       │
│  │   x₄ = RandomCrop(x, padding=4)                             │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Total: 5 augmented views                                                │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ For Each View: Compute Prediction                                        │
│                                                                           │
│  For i in [0, 1, 2, 3, 4]:                                               │
│                                                                           │
│    ┌─────────────────────────────────────────────────────────────┐       │
│    │ 1. Extract embedding                                        │       │
│    │    emb_i = model.get_embedding(x_i)                         │       │
│    │                                                             │       │
│    │ 2. Retrieve neighbors                                       │       │
│    │    dists_i, indices_i = memory_bank.search(emb_i, k=16)     │       │
│    │    neigh_emb_i = memory_bank.embeddings[indices_i]          │       │
│    │    neigh_labels_i = memory_bank.labels[indices_i]           │       │
│    │                                                             │       │
│    │ 3. Compute attention                                        │       │
│    │    attn_i, _ = model.attention(...)                        │       │
│    │                                                             │       │
│    │ 4. Weighted prediction                                     │       │
│    │    probs_i = attn_i @ one_hot(neigh_labels_i)              │       │
│    └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Collect: probs_list = [probs₀, probs₁, probs₂, probs₃, probs₄]        │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Average Predictions                                                       │
│                                                                           │
│  final_probs = mean(probs_list)  (10,)                                   │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Why TTA Works:                                                │       │
│  │                                                                │       │
│  │ 1. Variance Reduction                                         │       │
│  │    • Averages over multiple views                            │       │
│  │    • Reduces prediction variance                              │       │
│  │                                                                │       │
│  │ 2. Distribution Alignment                                     │       │
│  │    • Aligns test-time with training-time augmentations        │       │
│  │                                                                │       │
│  │ 3. Overconfidence Reduction                                  │       │
│  │    • Smooths extreme predictions                              │       │
│  │    • Improves calibration (ECE)                              │       │
│  │                                                                │       │
│  │ 4. Implicit Ensemble                                         │       │
│  │    • Multiple views provide ensemble effect                   │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Prediction:                                                              │
│    pred_class = argmax(final_probs)                                       │
│    confidence = max(final_probs)                                          │
└───────────────────────────────────────────────────────────────────────────┘

Results:
  Without TTA: ECE = 0.1300
  With TTA:    ECE = 0.0283  (78% reduction)
```

---

## 8. Comparison with Baselines

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Comparison: Attention vs Baselines                         │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ METHOD 1: Uniform kNN                                                    │
│                                                                           │
│  Neighbors: N₁, N₂, ..., Nₖ                                             │
│                                                                           │
│  Weights: αᵢ = 1/k  (equal weight for all neighbors)                    │
│                                                                           │
│  Prediction:                                                             │
│    probs = (1/k) × Σ one_hot(label_i)                                   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │ Example: k=5                                                │         │
│  │   Neighbors: [cat, cat, dog, bird, cat]                    │         │
│  │   Weights: [0.2, 0.2, 0.2, 0.2, 0.2]                       │         │
│  │   Result: [0.6, 0.2, 0.2, 0, 0, ...]  (60% cat)            │         │
│  └─────────────────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ METHOD 2: Distance-Weighted kNN                                          │
│                                                                           │
│  Neighbors: N₁, N₂, ..., Nₖ                                             │
│  Distances: d₁, d₂, ..., dₖ                                             │
│                                                                           │
│  Weights: αᵢ = softmax(-dᵢ / τ)                                          │
│           (closer neighbors get higher weight)                           │
│                                                                           │
│  Prediction:                                                              │
│    probs = Σ αᵢ × one_hot(label_i)                                      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │ Example: k=5                                                │         │
│  │   Neighbors: [cat, cat, dog, bird, cat]                    │         │
│  │   Distances: [0.1, 0.2, 0.3, 0.4, 0.5]                     │         │
│  │   Weights: [0.35, 0.28, 0.21, 0.10, 0.06]                  │         │
│  │   Result: [0.67, 0.21, 0.10, 0, 0, ...]  (67% cat)         │         │
│  └─────────────────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ METHOD 3: Attention-Weighted kNN (Ours)                                  │
│                                                                           │
│  Neighbors: N₁, N₂, ..., Nₖ                                             │
│  Distances: d₁, d₂, ..., dₖ                                             │
│  Labels: y₁, y₂, ..., yₖ                                                │
│                                                                           │
│  Weights: αᵢ = learned_attention(query, Nᵢ, dᵢ, yᵢ, ...)                │
│           (learned function considering multiple factors)                │
│                                                                           │
│  Attention considers:                                                     │
│    • Query-neighbor similarity (Q·K)                                     │
│    • Distance (learned distance bias)                                    │
│    • Label distribution (label-conditioned bias)                         │
│    • Prototype alignment (prototype-guided scoring)                        │
│                                                                           │
│  Prediction:                                                              │
│    probs = Σ αᵢ × one_hot(label_i)                                      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │ Example: k=5                                                │         │
│  │   Neighbors: [cat, cat, dog, bird, cat]                    │         │
│  │   Distances: [0.1, 0.2, 0.3, 0.4, 0.5]                     │         │
│  │   Learned Weights: [0.32, 0.30, 0.18, 0.12, 0.08]          │         │
│  │   Result: [0.70, 0.18, 0.12, 0, 0, ...]  (70% cat)         │         │
│  │                                                             │         │
│  │ Note: Weights learned to optimize classification loss      │         │
│  └─────────────────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE COMPARISON                                                    │
│                                                                           │
│  ┌──────────────┬───────────┬──────────┬──────────┬──────────┐          │
│  │ Method       │ Accuracy  │ ECE      │ NLL      │ F1-Macro │          │
│  ├──────────────┼───────────┼──────────┼──────────┼──────────┤          │
│  │ Uniform      │ 86.82%    │ 0.1297   │ 2.053    │ 86.78%   │          │
│  │ Distance     │ 86.85%    │ 0.1099   │ 2.054    │ 86.81%   │          │
│  │ Attention    │ 86.85%    │ 0.1300   │ 2.054    │ 86.81%   │          │
│  │ Attn + TTA   │ 87.20%    │ 0.0283   │ 0.967    │ 87.27%   │          │
│  └──────────────┴───────────┴──────────┴──────────┴──────────┘          │
│                                                                           │
│  Key Finding:                                                            │
│    • Attention alone ≈ Distance ≈ Uniform (marginal difference)         │
│    • TTA provides significant calibration improvement (78% ECE ↓)       │
│    • TTA is the key innovation, not attention mechanism                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Memory Bank Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Memory Bank (FAISS) Architecture                    │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ BUILDING THE MEMORY BANK                                                  │
│                                                                           │
│  Training Set: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}                       │
│    n = 50,000 samples                                                     │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Step 1: Extract Embeddings                                    │       │
│  │                                                                │       │
│  │ For each training sample:                                     │       │
│  │   emb_i = model.get_embedding(x_i)  (256,)                    │       │
│  │                                                                │       │
│  │ Result:                                                        │       │
│  │   embeddings: (n, 256) matrix                                 │       │
│  │   labels: (n,) array                                           │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                          │                                                 │
│                          ▼                                                 │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Step 2: Build FAISS Index                                     │       │
│  │                                                                │       │
│  │ Options:                                                       │       │
│  │                                                                │       │
│  │ 1. Flat L2 Index (for n < 100K):                             │       │
│  │    • Exact search                                              │       │
│  │    • O(n) search time                                          │       │
│  │    • Fast build time (~3ms for 50K)                          │       │
│  │                                                                │       │
│  │ 2. HNSW Index (for n > 100K):                                │       │
│  │    • Approximate search                                        │       │
│  │    • O(log n) search time                                     │       │
│  │    • Slower build time (~41s for 50K)                        │       │
│  │                                                                │       │
│  │ For CIFAR-10: Use Flat L2                                     │       │
│  └───────────────────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ QUERYING THE MEMORY BANK                                                  │
│                                                                           │
│  Query Embedding: q (256,)                                                │
│        │                                                                   │
│        ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ FAISS Search                                                   │       │
│  │                                                                │       │
│  │ dists, indices = index.search(q, k=16)                        │       │
│  │                                                                │       │
│  │ Returns:                                                       │       │
│  │   • dists: (16,) distances to nearest neighbors               │       │
│  │   • indices: (16,) indices in memory bank                     │       │
│  │                                                                │       │
│  │ Search Time: <0.05ms per query                                │       │
│  └───────────────────────────────────────────────────────────────┘       │
│        │                                                                   │
│        ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Extract Neighbor Information                                  │       │
│  │                                                                │       │
│  │ neigh_emb = embeddings[indices]  (16, 256)                    │       │
│  │ neigh_labels = labels[indices]  (16,)                         │       │
│  │ dists = dists  (16,)                                           │       │
│  └───────────────────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ MEMORY BANK UPDATES DURING TRAINING                                      │
│                                                                           │
│  Strategy: Periodic Updates                                              │
│                                                                           │
│  • Update frequency: Every N batches (configurable)                      │
│  • Reason: Embeddings change as model trains                             │
│  • Process:                                                              │
│      1. Recompute embeddings for all training samples                   │
│      2. Rebuild FAISS index                                              │
│      3. Continue training                                                │
│                                                                           │
│  Trade-off:                                                               │
│    • More frequent updates → more accurate neighbors                      │
│    • More frequent updates → slower training                              │
│    • Default: Update every epoch                                         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Loss Function Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Loss Function Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

Total Loss: L_total = L_knn + λ_con × L_contrastive + λ_ent × L_entropy

┌───────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 1: kNN Classification Loss                                      │
│                                                                           │
│  L_knn = CrossEntropy(log(probs), y_true)                                │
│                                                                           │
│  Where:                                                                   │
│    probs = attention @ one_hot(neighbor_labels)  (B, C)                  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Key Design:                                                    │       │
│  │                                                                │       │
│  │ Loss computed directly on attention-weighted predictions      │       │
│  │ This aligns training objective with evaluation                │       │
│  │                                                                │       │
│  │ Critical Fix (Experiment 3):                                   │       │
│  │   Before: Loss on classifier logits (WRONG)                   │       │
│  │   After: Loss on kNN predictions (CORRECT)                    │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Weight: λ_knn = 1.0 (primary loss)                                      │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 2: Contrastive Loss                                             │
│                                                                           │
│  L_contrastive = SupervisedContrastiveLoss(query_emb, y_true, margin)    │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Process:                                                        │       │
│  │                                                                │       │
│  │ 1. For each query embedding q_i:                              │       │
│  │                                                                │       │
│  │ 2. Find positives: same class                                 │       │
│  │    P = {j : y_j == y_i}                                       │       │
│  │                                                                │       │
│  │ 3. Find negatives: different class                             │       │
│  │    N = {j : y_j != y_i}                                       │       │
│  │                                                                │       │
│  │ 4. Compute similarities:                                       │       │
│  │    sim_pos = q_i · q_j for j in P                            │       │
│  │    sim_neg = q_i · q_j for j in N                            │       │
│  │                                                                │       │
│  │ 5. Contrastive loss:                                            │       │
│  │    L = -log(exp(sim_pos + margin) /                           │       │
│  │            (exp(sim_pos + margin) + Σ exp(sim_neg)))          │       │
│  │                                                                │       │
│  │ Margin: 0.5 (pushes negatives further apart)                  │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Weight: λ_con = 0.5                                                      │
│                                                                           │
│  Purpose: Learn discriminative embeddings                                │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 3: Entropy Regularization                                      │
│                                                                           │
│  L_entropy = -entropy(attention_weights)                                  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐       │
│  │ Entropy Computation:                                          │       │
│  │                                                                │       │
│  │  entropy = -Σ αᵢ × log(αᵢ)                                   │       │
│  │                                                                │       │
│  │  where αᵢ are attention weights                              │       │
│  │                                                                │       │
│  │  Maximum entropy: αᵢ = 1/k (uniform)                        │       │
│  │  Minimum entropy: αᵢ = 1 for one neighbor, 0 for others     │       │
│  └───────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  Weight: λ_ent = 0.01                                                     │
│                                                                           │
│  Purpose: Prevent attention collapse (encourage diverse attention)          │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ LOSS WEIGHT SUMMARY                                                       │
│                                                                           │
│  ┌─────────────────────┬──────────┬────────────────────────────┐       │
│  │ Component            │ Weight   │ Purpose                    │       │
│  ├─────────────────────┼──────────┼────────────────────────────┤       │
│  │ L_knn                │ 1.0      │ Primary classification     │       │
│  │ L_contrastive        │ 0.5      │ Embedding quality          │       │
│  │ L_entropy            │ 0.01     │ Attention diversity        │       │
│  └─────────────────────┴──────────┴────────────────────────────┘       │
│                                                                           │
│  Total: L_total = 1.0×L_knn + 0.5×L_contrastive + 0.01×L_entropy         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Summary of Novel Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Novel Components Summary                              │
└─────────────────────────────────────────────────────────────────────────┘

1. NEIGHBOR SELF-ATTENTION
   ────────────────────────
   • Neighbors attend to each other before query aggregation
   • Refines neighbor representations based on local structure
   • Enables context-aware neighbor refinement

2. LABEL-CONDITIONED ATTENTION BIAS
   ─────────────────────────────────
   • Learns how label distribution affects attention
   • Adapts attention based on neighbor label patterns
   • Handles skewed label distributions

3. PROTOTYPE-GUIDED SCORING
   ─────────────────────────
   • Learnable class prototypes (centroids)
   • Alignment score between neighbors and prototypes
   • Guides attention toward typical class examples

4. LEARNED TEMPERATURE PER HEAD
   ─────────────────────────────
   • Each attention head learns its own temperature
   • Clamped to [0.05, 2.0] for stability
   • Better than fixed temperature

5. END-TO-END TRAINING WITH kNN LOSS
   ───────────────────────────────────
   • Loss computed directly on attention-weighted predictions
   • Aligns training objective with evaluation
   • Critical for proper optimization

6. TEST-TIME AUGMENTATION (TTA)
   ─────────────────────────────
   • Averages predictions over multiple augmented views
   • Provides 78% ECE reduction
   • Key innovation for calibration improvement
```

---

_These diagrams provide comprehensive visual documentation of the Attn-KNN architecture, suitable for understating the project for new comers._
