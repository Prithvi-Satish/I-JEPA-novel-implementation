# QuadTree-JEPA: Hierarchical Multi-Scale Self-Supervised Vision Transformer

A high-performance implementation of **QuadTree Joint-Embedding Predictive Architecture (QuadTree-JEPA)** for fine-grained multi-scale visual representation learning.

---

## 🌟 Key Architectural Features (V2)

1. **Hybrid Relative-Variance & Quantile-Budget Tokenization**:
   - Contrast-invariant quadtree patch subdivision ($\sigma^2_{\text{patch}} / \sigma^2_{\text{image}}$) with top-$K$ priority allocation ($K = 256$ tokens) for deterministic GPU Tensor Core acceleration.
2. **Multi-Scale Z-Axis Fusion Bridge**:
   - Grouped Batched GEMM projections with 2D spatial + 1D scale sinusoidal position embeddings across Levels 0 ($64 \times 64$), 1 ($32 \times 32$), 2 ($16 \times 16$), and 3 ($8 \times 8$).
3. **Scale-Aware Attentive Pooling**:
   - Replaces uniform average pooling with learnable query cross-attention, ensuring fine-grained disease lesions are prioritized over coarse background foliage.
4. **Deep Residual Classifier Head**:
   - Non-linear 2-layer classifier head with LayerNorm, GELU, and Dropout (0.2).
5. **Class-Balanced Weighted Loss & Label Smoothing (0.1)**:
   - Mitigates dataset frequency imbalance across 18 classes and prevents overconfidence on ambiguous boundaries.
6. **VICReg Variance & Trace-Regularized Covariance Decorrelation**:
   - Prevents representation collapse and forces latent dimensions to be mutually orthogonal.

---

## 📁 Repository Structure

```
├── quadtree_jepa.py              # Core V2 Architecture (Tokenizer, Bridge, JEPA, Pooler, Classifier)
├── train_and_evaluate_cub.py    # Dedicated CUB-200-2011 Training & Evaluation Pipeline
├── train_and_evaluate_jepa.py   # Plant Pathology Training & Evaluation Pipeline
├── benchmark_label_efficiency.py # Label-Efficiency Benchmark (Supervised ViT vs. QuadTree-JEPA)
├── eval_knn.py                   # k-NN Feature Quality Benchmark
├── eval_benchmark_fast.py        # Fast Benchmark Evaluator over Cached Embeddings
├── finetune_fast.py              # Batched End-to-End Discriminative Fine-Tuning
├── download_cub_dataset.py       # CUB-200-2011 Dataset Downloader & Verifier (11.8k images)
├── download_plant_dataset.py     # Plant Pathology Dataset Downloader (18 Classes, 25k+ Images)
├── verify_modifications.py       # Comprehensive 10-Point Architecture Test Suite
├── test_full_pipeline.py         # Full Pipeline Dry-Run Integration Test
├── mods.md                       # Living Roadmap & Modifications Tracker (V1 & V2)
├── knowledgebank.md              # In-Depth Theoretical Reference & Context Bank
├── plots/                        # Generated figures, confusion matrices, and comparison plots
├── checkpoints/
│   ├── cub/                      # CUB-200 weights & feature caches
│   └── plant/                    # Plant pathology weights & feature caches
└── data/
    ├── cub_200_2011/             # CUB-200 bird species dataset
    └── plant_dataset/            # 18-class plant leaf dataset
```

---

## 🚀 Quickstart Workflows

### 1. Verify Pipeline Architecture
```bash
python verify_modifications.py
```

### 2. CUB-200-2011 Benchmark (Bird Species Multi-Scale Hierarchy)
```bash
# Download & verify CUB-200-2011 dataset (11,788 images across 200 species)
python download_cub_dataset.py

# Run Self-Supervised Pre-Training & Linear Probing
python train_and_evaluate_cub.py --pretrain_epochs 30 --probe_epochs 30
```

### 3. Plant Pathology Benchmark (18 Classes, 25,283 Images)
```bash
# Download plant dataset
python download_plant_dataset.py

# Run full pretraining and fine-tuning
python train_and_evaluate_jepa.py --pretrain_epochs 30 --probe_epochs 30 --finetune_epochs 20
```

### 4. Run Label Efficiency Benchmark
```bash
python benchmark_label_efficiency.py
```
