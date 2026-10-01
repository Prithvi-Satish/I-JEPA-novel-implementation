# QuadTree-JEPA: Project Context & Agent Quickstart Guide

> **Notice for any new AI Agent or Developer**:  
> Read this document first! It gives you a complete, up-to-date briefing of the architecture, current codebase state, verified real results, known environment constraints, and immediate next steps. Do not alter core verified numbers or re-introduce unverified baselines.

---

## 1. Executive Summary & Purpose

**QuadTree-JEPA** is a novel computer vision framework that unites **adaptive multi-scale spatial tokenization (QuadTree)** with self-supervised **Joint-Embedding Predictive Architectures (JEPA)** for fine-grained image representation.

### The Problem It Solves
Standard Vision Transformers (ViTs) discretize images into a uniform grid of fixed-size patches (e.g., $16 \times 16$). In fine-grained visual domains (e.g., botanical foliar disease lesions, avian plumage micro-structures):
- **Coarse uniform patches** dilute tiny discriminative lesions across large healthy regions.
- **Fine uniform patches** explode sequence length quadratically ($\mathcal{O}(N^2)$ memory and compute).
- **Classic QuadTrees** use adaptive recursive branching based on scalar thresholds (`.item()`), causing catastrophic host-device PCIe synchronization stalls and variable token counts that break dense GPU batch collation.

### The QuadTree-JEPA Solution
1. **Deterministic-Budget Vectorized QuadTree Tokenizer**: Partitions images into a 4-level spatial pyramid ($64\times64$ down to $8\times8$) with a fixed budget of exactly $K=256$ tokens per image, executed 100% on-device with zero CPU synchronization.
2. **Multi-Scale Z-Axis Fusion Bridge**: Maps multi-scale tokens to a uniform embedding dimension ($D=768$) using grouped GEMM projections, 2D continuous spatial encodings, and 1D learnable scale embeddings.
3. **Stochastic Bidirectional JEPA**: Self-supervised learning purely in latent space (no pixel reconstruction artifacts) by predicting fine targets from coarse context and vice-versa.
4. **Scale-Aware Attentive Pooling**: Aggregates multi-scale tokens with zero-initialized learnable level biases for downstream classification.

---

## 2. Verified Ground Truth Results vs. Fabricated Claims

> [!IMPORTANT]  
> **Documentation Integrity Rule**: Only report verified results backed by source logs. A prior agent populated `preprint/main.tex` with fabricated baselines (ResNet, Swin, ConvNeXt, I-JEPA comparisons, fake ablation numbers, and synthetic label efficiency tables). All fake numbers were audited and removed. Do **NOT** fabricate any experimental metrics.

| Metric / Experiment | Status | Verified Value | Source / Notes |
|---|---|---|---|
| **18-Class Plant Pathology (Fine-Tuned)** | **VERIFIED** | **90.65% Top-1 Acc, 88.22% Macro F1** | `mods.md` (lines 332, 346). 25,283 images, depth=6 ViT random-init, 30 SSL + 30 probe + 20 finetune epochs. |
| **V3 Pretrained Backbone (Target)** | **IN PROGRESS** | **Target: 93% – 96% Acc** | Depth=12 ViT-Base with ImageNet pretrained weights. Code implemented in `train_and_evaluate_jepa.py`. |
| **Frozen Linear Probe Acc** | **PENDING RUN** | To be recorded | Will be obtained during Phase 2A of the V3 training run. |
| **Label Efficiency (1%, 10%, 25%, 50%)** | **NOT YET RUN** | Script exists (`benchmark_label_efficiency.py`) | To be executed after V3 weights are trained. |
| **Ablation Studies (No bridge, no pyramid)** | **NOT YET RUN** | To be run if compute permits | Needs actual ablation training scripts. |
| **External Baselines (Swin, I-JEPA, ResNet)** | **EXPUNGED** | N/A | Removed from preprint; only report real baseline experiments. |

---

## 3. Core Architecture Specification (Verified in Code)

### Tokenizer Budget Allocation ($K = 256$)
For a $512 \times 512$ padded image ($8 \times 8 = 64$ root patches):
- **Level 0 ($64 \times 64$ px)**: Keep $N_0 = 20$ lowest-variance patches; split top-44 patches.
- **Level 1 ($32 \times 32$ px)**: $44 \times 4 = 176$ sub-patches. Keep $N_1 = 160$; split top-16 patches.
- **Level 2 ($16 \times 16$ px)**: $16 \times 4 = 64$ sub-patches. Keep $N_2 = 60$; split top-4 patches.
- **Level 3 ($8 \times 8$ px)**: $4 \times 4 = 16$ sub-patches. Keep all $N_3 = 16$.
- **Total Token Count**: $20 + 160 + 60 + 16 = 256$ tokens (constant tensor shape `(B, 256, D)`).

### Latent Dimension & Bridge
- **Patch Embedding Projections**: 4 separate Linear/Conv layers mapping $\{64^2\times3, 32^2\times3, 16^2\times3, 8^2\times3\} \to D=768$.
- **Positional Encoding**: Continuous 2D sinusoidal spatial coordinates + 1D learnable scale level embeddings ($0, 1, 2, 3$).
- **Predictor**: 3 Transformer blocks (1 cross-attention layer conditioning context tokens on target mask tokens + 2 self-attention layers).
- **EMA Target Encoder**: Exponential moving average teacher with cosine momentum decay ($0.996 \to 0.999$).
- **VICReg Anti-Collapse Regularizer**:
  $$\mathcal{L} = 25 \cdot \mathcal{L}_{\text{MSE}} + 25 \cdot \mathcal{L}_{\text{variance}} + 1.0 \cdot \mathcal{L}_{\text{covariance}}$$

---

## 4. Current State: V3 Pretrained Backbone Upgrade

The training script [`train_and_evaluate_jepa.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/train_and_evaluate_jepa.py) has been upgraded from V2 (depth=6 random-init) to **V3 (depth=12 ImageNet pre-trained ViT-Base)**:

1. **Backbone Architecture**:
   - `depth=12`, `heads=12`, `mlp_dim=3072`, `dim=768`, `dim_head=64`, `drop_path_rate=0.1`, `qkv_bias=True` (~86M params in ViT backbone, ~98M total with QuadTree components).
2. **Pre-trained Weight Transfer**:
   - Uses `load_pretrained_vit_weights(base_vit, model_name='vit_base_patch16_224')` via `timm`.
   - Transfers 100% of transformer layers (QKV projections, MLP layers, LayerNorms).
3. **Differential Learning Rates**:
   - Backbone learning rate: `2e-5` (protects established ImageNet representations from catastrophic forgetting).
   - Z-Bridge, Predictor, & Attentive Pooler: `2e-4` ($10\times$ higher to quickly adapt the randomly initialized 12.5M multi-scale parameters).
4. **Learning Rate Scheduler**:
   - 10-epoch linear warmup followed by cosine annealing decay (prevents large initial gradients from destabilizing the backbone).
5. **Loss & Momentum Tuning**:
   - Corrected VICReg alignment ($25\times$), variance ($25\times$), covariance ($1\times$).
   - Cosine-annealed EMA momentum from 0.996 to 0.999.

---

## 5. Critical Environment & Execution Rules (AGENT READ THIS)

> [!CAUTION]  
> **THE BACKGROUND TASK RUNNER HANGS ON `import torch` IN THIS ENVIRONMENT.**  
> When the AI agent attempts to run python scripts (`.venv\Scripts\python script.py`) via automated background tools, the process stalls for >3 minutes loading `torch_cuda.dll` or due to console output buffering.  
> **RULE**: **DO NOT run long training or CUDA scripts via agent background execution.** Always give the user clear, copy-pasteable PowerShell commands to run directly in their native terminal.

### Environment Specifications
- **Operating System**: Windows 11
- **Hardware**: Dedicated NVIDIA RTX GPU with **8GB VRAM**
- **Python Environment**: `c:\Users\Prithvi S\OneDrive\Documents\ALL PROJECTS\big dih shi\demo\vit-pytorch-main\.venv\Scripts\python.exe`
- **Installed Key Packages**: `torch`, `torchvision`, `timm>=0.9.0`, `albumentations`, `scikit-learn`, `pandas`, `tqdm`
- **Dataset Location**: `vit-pytorch-main/data/plant_dataset` (20,192 training images, 5,091 test images across 18 balanced classes).

---

## 6. Repository File Map

```
demo/
├── CONTEXT.md                       <- Root context pointer for new sessions
└── vit-pytorch-main/
    ├── CONTEXT.md                   <- Comprehensive project guide (this file)
    ├── train_and_evaluate_jepa.py   <- PRIMARY PLANT PIPELINE (Upgraded to V3)
    ├── smoke_test_v3.py             <- Quick GPU/VRAM/Weights sanity check script
    ├── quadtree_jepa.py             <- Core QuadTree + JEPA architecture (DO NOT BREAK)
    ├── benchmark_label_efficiency.py<- Evaluates probes at 1%, 10%, 25%, 50% data
    ├── download_plant_dataset.py    <- Downloads and prepares the 18-class dataset
    ├── mods.md                      <- Historical log of experiments and verified results
    ├── knowledgebank.md             <- Detailed mathematical and design rationale
    ├── vit_pytorch/
    │   └── vit.py                   <- Base ViT definition & timm weight loader
    └── preprint/
        ├── main.tex                 <- Academic paper draft (rewritten with honest facts)
        ├── references.bib           <- BibTeX citations
        └── preprint_audit.md        <- Audit trail of removed fabricated claims
```

---

## 7. Step-by-Step Execution Workflow

### Step 1: Run the V3 Smoke Test (User Terminal)
Before launching training, ensure the environment and weights load cleanly:
```powershell
cd "c:\Users\Prithvi S\OneDrive\Documents\ALL PROJECTS\big dih shi\demo\vit-pytorch-main"
.venv\Scripts\python smoke_test_v3.py
```
*Expected output*: Confirms CUDA availability, downloads `vit_base_patch16_224` (~340MB once), performs a forward pass with dummy tensor, verifies peak VRAM is well below 8GB.

### Step 2: Run V3 Pre-training and Evaluation (User Terminal)
```powershell
cd "c:\Users\Prithvi S\OneDrive\Documents\ALL PROJECTS\big dih shi\demo\vit-pytorch-main"
.venv\Scripts\python train_and_evaluate_jepa.py
```
*Pipeline Stages*:
1. **Phase 1 (SSL Pre-training)**: 30 epochs of self-supervised QuadTree-JEPA pre-training with differential LRs and VICReg regularizer. Saves checkpoint to `checkpoints/best_jepa_backbone.pth`.
2. **Phase 2A (Frozen Linear Probe)**: 30 epochs evaluating representation quality with frozen backbone.
3. **Phase 2B (End-to-End Fine-tuning)**: 20 epochs fine-tuning the entire network to achieve peak classification accuracy.

### Step 3: Run the Label Efficiency Benchmark (User Terminal)
Once pre-training is complete, evaluate semi-supervised label efficiency:
```powershell
.venv\Scripts\python benchmark_label_efficiency.py
```
This tests performance on 1%, 10%, 25%, and 50% labeled subsets to demonstrate JEPA representation power.

### Step 4: Update the Preprint (`preprint/main.tex`)
After real numbers are logged:
1. Update Table 1 with the real Frozen Probe accuracy and Fine-Tuned Top-1 / F1 scores.
2. Insert actual label efficiency numbers from `benchmark_label_efficiency.py`.
3. Recompile the LaTeX document.
