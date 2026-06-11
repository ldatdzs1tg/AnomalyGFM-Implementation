# AnomalyGFM — Paper Implementation & Extension

> **Paper:** *AnomalyGFM: Graph Foundation Model for Zero/Few-Shot Anomaly Detection*
> *(PDF included: `AnomalyGFM.pdf`)*

This repository is a **paper implementation** of AnomalyGFM, reproducing its results across the 11 benchmark datasets reported in the paper. Results obtained are consistent with the main result table in the paper.

Additionally, the model is **extended** by applying it to a new real-world dataset — **BERT4ETH** — an Ethereum transaction dataset for phishing account detection. Since BERT4ETH is tabular/transaction data (not natively a graph), a full graph construction pipeline was built from scratch before feeding into the same large-scale inference model used for T-Finance and T-Social.

---

## 📋 What AnomalyGFM Does

**AnomalyGFM** is a **Graph Foundation Model (GFM)** for node-level anomaly detection on attributed graphs. Its key insight is to train once on a labeled source graph, then generalize to unseen target graphs without retraining — achieving **zero-shot** cross-graph transfer.

It also supports **few-shot** adaptation, where a small number of labeled nodes from the target graph are used to fine-tune a lightweight prompt.

### Core Mechanism

1. **GCN Encoder** — 2-layer GCN encodes each node into a 300-dim embedding `h(v)`
2. **Residual Embedding** — captures how much a node deviates from its neighborhood:
   ```
   residual(v) = h(v) - mean_{u ∈ N(v)} h(u)
   ```
   Anomalous nodes tend to have large residuals (they differ from neighbors).
3. **Prototype Prompts** — two learnable vectors `p_n` (normal prototype) and `p_a` (abnormal prototype) serve as reference directions in embedding space.
4. **Anomaly Score** — cosine similarity of each node's residual against both prototypes:
   ```
   score(v) = exp(cos(residual_v, p_a)) + β × exp(-cos(residual_v, p_n))
   ```

### Training (Zero-Shot)

The model is trained **only on the source graph** (default: Facebook) with a combined loss:
```
L = L_BCE + α × L_align

L_align = mean(||residual_a - p_a||) + 0.1 × mean(||residual_n - p_n||)
```
- `L_BCE`: Supervised binary cross-entropy on labeled train/val nodes of the **source** graph.
- `L_align`: Pulls anomalous node residuals toward `p_a`, and (with lower weight 0.1) normal residuals toward `p_n`.

**Adjacency convention used:** `A = A_norm + I` — different from the standard GCN `(A+I)_norm`.

---

## 🗂️ Repository Structure

```
AnomalyGFM/
│
├── AnomalyGFM.pdf                        # Original paper
│
├── ReproduceCode/                        # Paper reproduction experiments
│   ├── svd_trans.py                      # Step 0: Truncated SVD — unify feature dimension to d'=10
│   │
│   ├── zero-shot/                        # Zero-shot: train on source, evaluate on all targets
│   │   ├── model.py                      # AnomalyGFM (GCN + ResidualComputer + LossFunction)
│   │   ├── main.py                       # Full training + evaluation loop
│   │   └── utils.py                      # load_mat, load_train_mat, load_test_mat, normalize_adj, ...
│   │
│   ├── few-shot/                         # Few-shot: freeze GCN, fine-tune prompt only
│   │   ├── model.py                      # normalFewshotModel / abnormalFewshotModel + SimplePrompt
│   │   ├── main_normal.py                # Fine-tune with K labeled normal nodes
│   │   ├── main_abnormal.py              # Fine-tune with K labeled abnormal nodes
│   │   ├── utils.py                      # Data loading helpers
│   │   └── model_weights_300.pth         # Pre-trained zero-shot checkpoint (d=300)
│   │
│   ├── large-scale/                      # Scalable inference for T-Finance & T-Social
│   │   ├── model.py                      # Batched GCN — input: [B, S, D] subgraph tensors
│   │   ├── fast_inference_main.py        # Batched inference pipeline
│   │   ├── model_residual2.pth           # Pre-trained checkpoint
│   │   └── utils.py                      # normalize_adj
│   │
│   └── hyperparameter comparison/        # Ablation studies
│       ├── ALPHA_performance.py          # Sweep α ∈ {0.6, 0.8, 1.0, 1.2} over 9 datasets
│       ├── T_performance.py              # Sweep embedding dim T over datasets
│       ├── ALPHA_results.csv             # Recorded AUROC/AUPRC per α
│       ├── T_results.csv                 # Recorded AUROC/AUPRC per T
│       ├── visualization.py              # Plot sweep results
│       ├── model.py / utils.py           # Shared with zero-shot
│       ├── alpha_figure.png              # α sensitivity plot
│       └── T_figure.png                  # T sensitivity plot
│
├── datasets for AnomalyGFM/             # All 11 benchmark datasets (SVD-preprocessed)
│   ├── Facebook_svd.mat                 # Source (training) graph
│   ├── Amazon_upu_svd.mat
│   ├── amazon_svd.mat
│   ├── Disney_svd.mat
│   ├── elliptic_svd.mat
│   ├── questions_svd.mat
│   ├── reddit_svd.mat
│   ├── t_finance_svd.mat
│   ├── tolokers_svd.mat
│   ├── yelp_svd.mat
│   ├── tfinance_feature_5_1.npy         # Pre-sampled subgraph tensors (large-scale inference)
│   ├── tfinance_label_5_1.npy
│   ├── tsocial_feature_7_1.npy
│   ├── tsocial_label_7_1.npy
│   └── dts_info.xlsx / dts_info.pdf     # Dataset statistics reference
│
└── b4e/                                  # BERT4ETH extension — see b4e/README.md
    ├── README.md                         # Full B4E documentation
    ├── data_raw/                         # Raw input (RAR + phisher list)
    │   ├── b4e_dataset.rar               # Extract before running!
    │   ├── phisher_account.txt
    │   └── README.md
    ├── data_structured/                  # Generated by preprocess.ipynb
    │   ├── b4e.mat
    │   ├── b4e_svd_10.mat
    │   ├── b4e_feature_5_1.npy
    │   └── b4e_label_5_1.npy
    └── src/                              # Source code
        ├── preprocess.ipynb              # Step 1
        ├── fast_inference_main.py        # Step 2
        ├── model.py / utils.py
        └── model_residual2.pth

result/                                   # Experiment result tables & figures
    ├── Zero-shot Table.xlsx
    ├── Large-scale Table.xlsx
    ├── Normal Few-shot Table.xlsx
    ├── Abnormal Few-shot.xlsx
    └── *.png                             # Visualization plots
```

---

## 📊 Datasets

All 11 benchmark datasets are preprocessed with **Truncated SVD** to a unified feature dimension `d' = 10` before training/evaluation. This is a key step from the paper that enables cross-graph transfer despite heterogeneous original feature spaces.

| # | Dataset | Scale | Anomaly Type | β |
|---|---|---|---|---|
| 1 | Facebook | Small | Fake accounts | 0.0 |
| 2 | Amazon_upu | Small | Fake reviews | 0.0 |
| 3 | Amazon | Small | Fake reviews | 0.0 |
| 4 | Disney | Small | Bot accounts | 0.0 |
| 5 | Reddit | Small | Spam accounts | 0.0 |
| 6 | Tolokers | Medium | Rule violations | 0.0 |
| 7 | Yelp | Medium | Fake reviews | 0.0 |
| 8 | Questions | Medium | Fake Q&A | 0.0 |
| 9 | Elliptic | Large | Illicit transactions | **4.0** |
| 10 | T-Finance | Large | Financial fraud | **4.0** |
| 11 | T-Social | Very large | Social spam | 0.0 |

> **β heuristic:** `β = 4.0` for datasets with average intra-graph edge cosine similarity < 0.5 (heterophilous); `β = 0.0` (equivalent to using only the abnormal prototype score) for the rest.

The **source (training) graph is Facebook**. All other 10 datasets are **zero-shot test targets** — the model never trains on them.

---

## 🏗️ Model Architecture

### Zero-Shot Model (`AnomalyGFM`)

```
Input: Node features (N, d'=10)  ←  after SVD unification
         │
   GCNLayer 1: Linear(d' → 300) + PReLU  [Xavier init, no bias on weights]
         │
   GCNLayer 2: Linear(300 → 300) + PReLU
         │
   ┌─────┴──────────────────────────────────┐
   │                                        │
ResidualComputer                    Linear Classifier
h(v) - scatter_mean(h[N(v)])        logit → L_BCE
   │
Prototype MLPs
  fc_normal_prompt:   Linear(300,300) + ReLU  →  p_n
  fc_abnormal_prompt: Linear(300,300) + ReLU  →  p_a
   │
Anomaly Score = exp(cos(residual, p̂_a)) + β·exp(-cos(residual, p̂_n))
```

### Few-Shot Extension (`SimplePrompt`)

After zero-shot pre-training, **all GCN parameters are frozen**. Only the prompt adapter is fine-tuned:

```python
SimplePrompt.forward(x):
    return x + ReLU(Linear(x)) + global_emb   # learnable residual + global shift
```

- **`main_normal.py`** — uses K labeled *normal* nodes; minimizes `||residual - p_n||`
- **`main_abnormal.py`** — uses K labeled *abnormal* nodes; minimizes `||residual - p_a||`

### Large-Scale Model (Batched Subgraph Inference)

For T-Finance and T-Social (millions of nodes), full-graph GCN is infeasible. Instead:

1. Each node is represented as a small **sampled subgraph** of size `S` (S-1 neighbors + the target node at position `[:, -1, :]`)
2. The model operates on batched `[B, S, D]` tensors using `torch.bmm`
3. At inference, an **identity adjacency matrix** is used — nodes within a subgraph are processed independently, avoiding over-smoothing in the small-subgraph regime

---

## 🔬 BERT4ETH Extension

### Folder Structure

> **⚠️ Before running:** Extract `b4e/data_raw/b4e_dataset.rar` into `b4e/data_raw/`. See `b4e/data_raw/README.md` for instructions.

```
b4e/
├── data_raw/                   ← Extract b4e_dataset.rar here first
│   ├── b4e_dataset.rar
│   ├── phisher_account.txt
│   ├── normal_eoa_transaction_in_slice_1000K.csv   (after extraction)
│   ├── normal_eoa_transaction_out_slice_1000K.csv  (after extraction)
│   ├── phisher_transaction_in.csv                  (after extraction)
│   └── phisher_transaction_out.csv                 (after extraction)
├── data_structured/            ← Generated files (output of preprocess)
│   ├── b4e.mat
│   ├── b4e_svd_10.mat
│   ├── b4e_feature_5_1.npy
│   └── b4e_label_5_1.npy
└── src/                        ← All source code
    ├── preprocess.ipynb          ← Step 1: Run all cells
    ├── fast_inference_main.py    ← Step 2: Run inference
    ├── model.py
    ├── utils.py
    └── model_residual2.pth
```

### Motivation

BERT4ETH is an Ethereum blockchain transaction dataset originally used for phishing account detection. Unlike the 11 paper datasets which already come as graphs (adjacency + node features), BERT4ETH consists of raw **transaction logs** (tabular data). A complete graph construction pipeline was built to transform this data into a format compatible with the AnomalyGFM large-scale inference setting.

### Pipeline (run `b4e/src/preprocess.ipynb`)

```
Raw Ethereum transaction CSVs (IN/OUT)
         │
  load_data()
  ├── Parse all transactions
  └── Filter out records with empty from/to addresses
         │
  seq_duplicate()
  └── Merge transactions between the same address pair within 3-day windows
         │
  generate_bitcoin_mat()                     →  b4e.mat
  ├── Node discovery: all unique addresses found in transaction history
  ├── Graph topology: directed edges u→v for each transaction direction
  ├── Feature extraction: 28 handcrafted features per node (see below)
  ├── Preprocessing: log1p on skewed cols, signed-log on balance, RobustScaler, clip[-5,5]
  └── Label mapping: match addresses against phisher_account.txt  →  Label: 0/1
         │
  generate_svd_mat()                         →  b4e_svd_10.mat
  └── TruncatedSVD: 28-dim → 10-dim (same d' as benchmark datasets)
      + second RobustScaler applied post-SVD for stability
         │
  generate_npy_for_fast_inference()          →  b4e_feature_5_1.npy + b4e_label_5_1.npy
  ├── DGL random walk (length=2S) to sample S-1 neighbors per node
  ├── Build [N, S, D] tensor — target node always last
  └── Isolated nodes self-pad with their own features
         │
  fast_inference_main.py  (b4e/src/)
  └── Same batched inference as large-scale/  →  AUROC, AUPRC, Precision@K, Recall@K
```

### 28 Handcrafted Node Features

| Category | Features |
|---|---|
| **Volume (9)** | total_val_in/out, avg_val_in/out, avg_tx_val_in/out, balance, turnover_rate, net_flow_ratio |
| **Count & Degree (8)** | total_tx_in/out, unique_src/dst_addresses, tx_per_address, in_out_addr_ratio, addr_flow_ratio, addr_reuse_rate |
| **Temporal (5)** | lifespan_days, avg_hours_between_tx, std_hours_between_tx, tx_per_hour, overall_val_velocity |
| **Aggregation (6)** | total_clusters_in/out, tx_per_clusters, avg_clusters_in/out, max_clusters_size |

> *"Clusters"* refer to groups of transactions merged by the 3-day window aggregation step.

### Evaluation Metrics (B4E only)

Because phishing accounts are extremely rare, standard AUC/AP are supplemented with top-K precision/recall metrics:

| Metric | Description |
|---|---|
| **AUROC** | Area under ROC curve |
| **AUPRC** | Area under Precision-Recall curve |
| **Precision@0.1%** | Fraction of true phishers in top 0.1% scored nodes |
| **Recall@0.1%** | Fraction of all phishers captured in top 0.1% |
| **Precision@0.05%** | Same at top 0.05% |
| **Recall@0.05%** | Same at top 0.05% |

---

## 🚀 Reproducing the Paper Results

### 0. Install Dependencies

Tested environment: **Python 3.8**, **PyTorch 2.1.0**, **CUDA 12.1** (CPU also supported).

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** `requirements.txt` installs CPU builds by default. For GPU, reinstall `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse`, and `dgl` with the CUDA-specific wheels from their official pages:
> [pytorch.org](https://pytorch.org/get-started/locally/) · [pyg.org](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) · [dgl.ai](https://www.dgl.ai/pages/start.html)

### 1. SVD Preprocessing

All `.mat` files in `datasets for AnomalyGFM/` are already SVD-preprocessed (suffix `_svd`). To re-run SVD on a raw `.mat` file:

```bash
python ReproduceCode/svd_trans.py
# Edit dataset_str and emb_dimension=10 inside the script
```

### 2. Zero-Shot Evaluation (9 small/medium/large datasets)

```bash
cd ReproduceCode/zero-shot
python main.py
```

Key config at top of `main.py`:
```python
DATASET_TRAIN = "Facebook_svd"   # Source: trained once, never changes
DATASET_TEST  = "Amazon_upu_svd" # Target: any of the 9 test datasets
HIDDEN_DIM    = 300
OUTPUT_DIM    = 300
LR            = 1e-4
ALPHA         = 1.0
EPOCHS        = 300
BETA          = 0.0   # Set to 4.0 for elliptic_svd and t_finance_svd
```

Metrics reported every 50 epochs: AUROC and AUPRC on the test dataset.

### 3. Large-Scale Zero-Shot (T-Finance, T-Social)

```bash
cd ReproduceCode/large-scale
python fast_inference_main.py
# Set DATASET = "tfinance" or "tsocial" and SUBGRAPH_SIZE accordingly
```

Input files `{DATASET}_feature_{S+1}_1.npy` and `{DATASET}_label_{S+1}_1.npy` must exist in `datasets for AnomalyGFM/`.

### 4. Few-Shot with Normal Examples

```bash
cd ReproduceCode/few-shot
python main_normal.py
# DATASET_TEST = target, K_SHOT = 1 (default)
```

### 5. Few-Shot with Abnormal Examples

```bash
cd ReproduceCode/few-shot
python main_abnormal.py
# DATASET_TEST = target, K_SHOT = 10 (default)
```

### 6. BERT4ETH Application

```bash
# Step 0: Extract data_raw/b4e_dataset.rar into data_raw/  (see b4e/data_raw/README.md)

# Step 1: Run all cells in b4e/src/preprocess.ipynb
#         Generates data_structured/b4e_feature_5_1.npy and b4e_label_5_1.npy

# Step 2: Run inference
cd b4e/src
python fast_inference_main.py
```

> 📖 See [`b4e/README.md`](b4e/README.md) for full pipeline documentation.

---

## ⚙️ Hyperparameter Reference

| Hyperparameter | Default | Role |
|---|---|---|
| `HIDDEN_DIM` / `OUTPUT_DIM` | 300 | GCN hidden & embedding dimension |
| `LR` | 1e-4 | Adam optimizer learning rate |
| `ALPHA` (α) | 1.0 | Weight of alignment loss `L_align`; swept over {0.6, 0.8, 1.0, 1.2} in ablation |
| `BETA` (β) | 0.0 or 4.0 | Score balance between abnormal and normal prototype signals |
| `K_SHOT` | 1 (normal) / 10 (abnormal) | Labeled examples for few-shot prompt tuning |
| `EPOCHS` | 300 (zero-shot) / 5–20 (few-shot) | Training epochs |
| `SUBGRAPH_SIZE` (S) | 6 (T-Social) / 4 (T-Finance, B4E) | Neighborhood subgraph size for fast inference |
| `BATCH` | 32768 | Inference batch size |
| `emb_dimension` (d') | 10 | SVD output dimension — unified across all datasets |

---

## 🔗 References

- **Paper:** AnomalyGFM: Graph Foundation Model for Zero/Few-Shot Anomaly Detection (`AnomalyGFM.pdf`)
- **Datasets:** All 11 benchmark datasets are sourced directly from the AnomalyGFM paper and its supplementary materials.
- **BERT4ETH:** Ethereum phishing transaction dataset used as the extension target.
