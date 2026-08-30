# QuadTree-JEPA Project: Modifications & Roadmap (`mods.md`)

This living document tracks planned, in-progress, and completed modifications for the QuadTree-JEPA framework and benchmark pipeline.

---

## 📋 Status Overview

- **Completed**: [MOD-01], [MOD-02], [MOD-04]
- **Pending/Next**: [MOD-03] (Expanded Dataset), [MOD-05] (3-Way Benchmark Comparison)
- **Last Updated**: 2026-08-30

---

## 🛠️ Modifications Directory

### [MOD-01] Multi-Scale Token Utilization (Levels 0, 1, 2, 3)
* **Status**: ✅ **Implemented**
* **Target Files**:
  * [`quadtree_jepa.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/quadtree_jepa.py)
* **What was changed**:
  1. **Pre-training (`forward`)**: Target queries now include both Level 2 ($16 \times 16$) and Level 3 ($8 \times 8$) tokens for fine-grained lesion learning.
  2. **Feature Extraction (`extract_features`)**: Now projects and passes all multi-scale tokens (0, 1, 2, 3) through `ZAxisFusionBridge` and `context_encoder` with global mean pooling so downstream classifiers receive fine lesion details.

---

### [MOD-02] End-to-End Fine-Tuning with Discriminative Learning Rates
* **Status**: ✅ **Implemented**
* **Target Files**:
  * [`quadtree_jepa.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/quadtree_jepa.py)
  * [`train_and_evaluate_jepa.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/train_and_evaluate_jepa.py)
* **What was changed**:
  1. Created [`QuadtreeClassifier`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/quadtree_jepa.py#L218-L250) module wrapper.
  2. Implemented `train_finetuned_classifier()` and `evaluate_finetuned_model()` in `train_and_evaluate_jepa.py` (Phase 2B & 3B).
  3. Configured discriminative parameter groups:
     * **Backbone (`context_encoder` + `z_bridge`)**: $1 \times 10^{-5}$ (gentle feature adaptation without catastrophic forgetting)
     * **Classification Head (`nn.Linear`)**: $1 \times 10^{-3}$ (rapid class boundary learning)

---

### [MOD-04] Batch-Level Variance Regularization & Stability
* **Status**: ✅ **Implemented**
* **Target Files**:
  * [`train_and_evaluate_jepa.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/train_and_evaluate_jepa.py)
* **What was changed**:
  1. Modified `run_pretraining` to buffer target representation tokens across `GRAD_ACCUM_STEPS` (effective batch size of 16 images).
  2. Variance loss is now computed across the full cross-sample token batch rather than within single images, stabilizing the latent space against dimensional collapse.

---

### [MOD-03] Dataset Expansion & Unlabeled SSL Pretraining Scaling
* **Status**: ⏳ **Awaiting Expanded Dataset**
* **Target Files**:
  * [`download_plant_dataset.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/download_plant_dataset.py)
* **Guidance**:
  * 1,000–2,000 images minimum.
  * 5,000–10,000+ images recommended for optimal SSL representations.

---

### [MOD-05] 3-Way Label Efficiency Benchmark (Supervised vs Probe vs Fine-Tuned)
* **Status**: 📝 **Planned for Final Step**
* **Target Files**:
  * [`benchmark_label_efficiency.py`](file:///c:/Users/Prithvi%20S/OneDrive/Documents/ALL%20PROJECTS/big%20dih%20shi/demo/vit-pytorch-main/benchmark_label_efficiency.py)
* **Plan**:
  * Add the fine-tuned classifier curve alongside Supervised ViT and Frozen Probe across label budgets (5, 10, 20, 40, 80 samples/class).

---

## 📌 Change Log / History

| Date | Mod ID | Description | Status |
| :--- | :--- | :--- | :--- |
| 2026-08-30 | MOD-01 | Multi-scale token utilization in forward pass and feature extraction | ✅ Completed |
| 2026-08-30 | MOD-02 | Discriminative fine-tuning module & dual evaluation pipeline | ✅ Completed |
| 2026-08-30 | MOD-04 | Batch-level variance regularization across gradient accumulation window | ✅ Completed |
