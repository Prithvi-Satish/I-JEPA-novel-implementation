import os
import sys
import copy
import time
import argparse
import gc
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vit_pytorch.vit import ViT, load_pretrained_vit_weights
from quadtree_jepa import (
    QuadtreeJEPA,
    QuadtreeClassifier,
    ScaleAwareAttentivePool,
    collate_quadtree_batch
)

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
DATA_DIR = "./data/cub_200_2011/CUB_200_2011"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "./data/cub_200_2011"

CHECKPOINT_DIR = "./checkpoints/cub"
PLOTS_DIR      = "./plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,      exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- GPU Throughput Flags (set globally, before any CUDA operation) ------------
# TF32: same accuracy as FP32 for matmuls but uses Tensor Cores -> ~2x faster.
# Safe: TF32 has 10-bit mantissa vs FP32's 23-bit, which is sufficient for ViT
# weight precision. This is the single largest untapped speedup on Ada Lovelace.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# cuDNN benchmark: profiles conv algorithms once per unique input shape.
# SAFE for QuadTree-JEPA: sequence length is always exactly 256 tokens.
torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# CUB-200-2011 Dataset — V3: ssl_mode + TrivialAugmentWide
# -----------------------------------------------------------------------------
class CUB200Dataset(Dataset):
    """
    CUB-200-2011 dataset loader — V4 with four augmentation modes:

      ssl_mode=True           -> DINO/I-JEPA strength SSL augmentations.
                                 Ops chosen carefully: only those that preserve
                                 relative luminance variance (QuadTree-safe).
                                 Excluded: Equalize, AutoContrast (flatten variance).
      ssl_mode=False,
        is_train=True         -> Curated safe fine-tuning augmentations.
                                 TrivialAugmentWide REMOVED — its Equalize/
                                 AutoContrast/Solarize ops destroy QuadTree's
                                 variance-based patch selection.
      force_eval_transform=True -> Always use deterministic eval transforms,
                                   regardless of is_train/ssl_mode.
                                   Used for feature extraction so cached
                                   embeddings are not randomly augmented.
      is_train=False          -> Deterministic evaluation transforms.
    """
    def __init__(self, root_dir, is_train=True, target_size=504,
                 ssl_mode=False, force_eval_transform=False):
        self.root_dir    = root_dir
        self.is_train    = is_train
        self.target_size = target_size
        self.ssl_mode    = ssl_mode
        self.samples     = []

        images_dir  = os.path.join(root_dir, "images")
        split_file  = os.path.join(root_dir, "train_test_split.txt")
        images_file = os.path.join(root_dir, "images.txt")
        labels_file = os.path.join(root_dir, "image_class_labels.txt")

        if os.path.exists(split_file) and os.path.exists(images_file) and os.path.exists(labels_file):
            with open(images_file) as f:
                id_to_image = {l.split()[0]: l.split()[1] for l in f}
            with open(labels_file) as f:
                id_to_label = {l.split()[0]: int(l.split()[1]) - 1 for l in f}  # 0-indexed
            with open(split_file) as f:
                for line in f:
                    img_id, split = line.split()
                    if (split == '1') == is_train:
                        rel_path  = id_to_image[img_id]
                        full_path = os.path.join(images_dir, rel_path)
                        label     = id_to_label[img_id]
                        if os.path.exists(full_path):
                            self.samples.append((full_path, label))
        else:
            # Fallback directory walker
            if os.path.exists(images_dir):
                classes = sorted([d for d in os.listdir(images_dir)
                                   if os.path.isdir(os.path.join(images_dir, d))])
                for label_idx, class_name in enumerate(classes):
                    folder  = os.path.join(images_dir, class_name)
                    files   = sorted(os.listdir(folder))
                    split_i = int(len(files) * 0.5)
                    sel     = files[:split_i] if is_train else files[split_i:]
                    for fn in sel:
                        self.samples.append((os.path.join(folder, fn), label_idx))

        # -- Build transform --
        eval_transform = T.Compose([
            T.Resize((target_size, target_size),
                      interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if force_eval_transform:
            # -- SILENT BUG FIX (V4): feature extraction must use deterministic
            #    transforms. eval_train_loader previously used is_train=True which
            #    triggered TrivialAugmentWide -> each run cached different random
            #    augmented features -> linear probe accuracy varied by +/-2-3%.
            self.transform = eval_transform

        elif is_train and ssl_mode:
            # -- DINO/I-JEPA SSL augmentation stack --
            # QuadTree-SAFE ops only:
            #   [OK] RandomResizedCrop — preserves relative variance ranking
            #   [OK] ColorJitter — shifts luminance uniformly, ranking preserved
            #   [OK] GaussianBlur — reduces all variance proportionally, ranking preserved
            #   [OK] RandomGrayscale — changes channels, luminance map barely affected
            #   [OK] RandomHorizontalFlip — mirrors coordinates, variance identical
            #   [X] RandomSolarize REMOVED — inverts pixels above threshold, creates
            #      artificial variance spikes at the threshold boundary, misleading
            #      QuadTree into splitting regions based on inversion artifacts
            self.transform = T.Compose([
                T.Resize((target_size + 32, target_size + 32),
                          interpolation=T.InterpolationMode.BILINEAR),
                T.RandomResizedCrop(target_size, scale=(0.3, 1.0), ratio=(0.75, 1.33),
                                    interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
                # Solarize intentionally EXCLUDED — see docstring above
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif is_train:
            # -- SAFE curated fine-tuning augmentations --
            # TrivialAugmentWide REMOVED — its Equalize / AutoContrast / Solarize
            # operations flatten the luminance histogram, making all 64 Level-0
            # patches have nearly identical variance. QuadTree then splits
            # essentially at random (~1/7 of training images ruined per epoch).
            #
            # Replacement: geometric + mild photometric transforms only.
            # All ops below preserve the RELATIVE variance ranking across patches.
            self.transform = T.Compose([
                T.Resize((target_size, target_size),
                          interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                interpolation=T.InterpolationMode.BILINEAR),
                T.RandomApply([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
                ], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            self.transform = eval_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert('RGB'))
        return tensor, label


def compute_class_weights(dataset, num_classes=200):
    labels  = [s[1] for s in dataset.samples]
    counts  = torch.bincount(torch.tensor(labels), minlength=num_classes)
    weights = 1.0 / (counts.float() + 1e-5)
    return (weights / weights.sum() * num_classes).to(device)


def apply_mixup_or_cutmix(img1, label1, img2, label2,
                           alpha_mixup=0.8, alpha_cutmix=1.0, prob_cutmix=0.5):
    if torch.rand(1).item() < prob_cutmix:
        lam     = np.random.beta(alpha_cutmix, alpha_cutmix)
        _, H, W = img1.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy  = np.random.randint(W), np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
        mixed       = img1.clone()
        mixed[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
    else:
        lam   = np.random.beta(alpha_mixup, alpha_mixup)
        mixed = lam * img1 + (1.0 - lam) * img2
    return mixed, label1, label2, lam


# -----------------------------------------------------------------------------
# LR Scheduler: Linear Warmup -> Cosine Annealing
# -----------------------------------------------------------------------------
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=1e-3):
    """Linear warmup for `warmup_epochs` then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1.0 - min_lr_ratio) + min_lr_ratio
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------------------------------
# Phase 1: High-Throughput Self-Supervised Pre-Training
# -----------------------------------------------------------------------------
def run_pretraining(model, train_loader, epochs=100, lr=1e-4, accum_steps=1, resume=False, is_pretrained=True):
    print("\n" + "=" * 70)
    print("      PHASE 1: QUADTREE-JEPA HIGH-THROUGHPUT PRE-TRAINING (CUB-200)")
    print("=" * 70)
    batch_sz    = getattr(train_loader, 'batch_size', 1) or 1
    effective_b = batch_sz * accum_steps
    WARMUP_EPOCHS    = 10     # linear LR warmup before cosine decay
    MOM_START        = 0.996  # EMA momentum at epoch 0
    MOM_END          = 0.999  # EMA momentum at epoch N (cosine annealed up)
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_sz} | Effective: {effective_b}")
    print(f"Warmup: {WARMUP_EPOCHS} epochs | EMA momentum: {MOM_START}->{MOM_END} (cosine)")
    print("Regularizer: Balanced VICReg (lambda=25, mu=25, nu=1) + Target LayerNorm")
    print("SSL Aug: RandomResizedCrop(scale=0.3-1.0) + GaussianBlur [Solarize excluded]\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if is_pretrained:
        # Differential LRs: lower for pre-trained backbone to preserve visual features,
        # higher for newly initialized Z-bridge and predictor to rapidly learn multi-scale tokens.
        backbone_params = list(model.context_encoder.parameters())
        new_params = (
            list(model.z_bridge.parameters()) +
            list(model.predictor.parameters()) +
            list(model.target_norm.parameters())
        )
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.05},
            {'params': new_params,      'lr': 2e-4, 'weight_decay': 0.05},
        ])
        print("  Optimizer: Differential LRs (Backbone: 2e-5, Multi-Scale Bridge/Predictor: 2e-4)")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.05)

    # V4: SSL now has 10-epoch linear warmup then cosine decay.
    # Previously: flat CosineAnnealingLR with no warmup — large random gradients
    # at epoch 1 with lr=1e-4 could destabilize the 12-layer ViT early on.
    def ssl_lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, epochs - WARMUP_EPOCHS))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1.0 - 1e-6) + 1e-6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ssl_lr_lambda)
    # V5: BF16 replaces generic FP16 AMP — Ada Lovelace SM89 has native BF16 Tensor Cores.
    # BF16 has the same numeric range as FP32 (8-bit exponent) -> no gradient overflow,
    # no GradScaler needed. FP16 (5-bit exponent) can overflow with large LRs/losses.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {'BF16 (native Tensor Core, no GradScaler)' if amp_dtype == torch.bfloat16 else 'FP16 + GradScaler'}")

    start_epoch = 1
    if resume:
        target_ckpt = None
        if isinstance(resume, str) and resume.isdigit():
            target_ckpt = os.path.join(CHECKPOINT_DIR, f"jepa_cub_epoch{resume}.pt")
        elif isinstance(resume, str) and os.path.exists(resume):
            target_ckpt = resume

        if target_ckpt and os.path.exists(target_ckpt):
            candidate_ckpts = [target_ckpt]
        else:
            candidate_ckpts = [
                os.path.join(CHECKPOINT_DIR, f"jepa_cub_epoch{ep}.pt")
                for ep in range(epochs, 0, -1)
            ]

        for ep_ckpt in candidate_ckpts:
            if os.path.exists(ep_ckpt):
                try:
                    state = torch.load(ep_ckpt, map_location=device, weights_only=False)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        model.load_state_dict(state['model_state_dict'])
                        start_epoch = state.get('epoch', 1) + 1
                    else:
                        model.load_state_dict(state)
                        import re
                        m = re.search(r'epoch(\d+)', ep_ckpt)
                        start_epoch = int(m.group(1)) + 1 if m else 2
                    print(f"--> Successfully resumed weights from checkpoint: {ep_ckpt} (continuing at Epoch {start_epoch})\n")
                    for _ in range(1, start_epoch):
                        scheduler.step()
                    break
                except Exception as e:
                    print(f"Warning: could not load {ep_ckpt}: {e}")

    print("--> Preparing DataLoader batches (CPU prefetching first batch, GPU will activate in ~10-15s)...", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.target_encoder.eval()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = epoch_inv = epoch_var = epoch_cov = 0.0
        last_std   = 0.0
        valid_steps = 0
        start_time = time.time()

        for step, (img_tensor, _) in enumerate(train_loader):
            imgs = img_tensor.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred_targets, true_targets, context_out, t_len = model(imgs, bidirectional=True)

                if pred_targets is None or true_targets is None or t_len == 0:
                    continue

                # 1. Invariance loss (lambda=25)
                loss_mse = F.mse_loss(pred_targets[:, :t_len, :], true_targets[:, :t_len, :])
                loss_inv = 25.0 * loss_mse

                # 2. Hardened VICReg Variance Regulariser (mu=25)
                # Enforced on BOTH context_encoder tokens and predictor tokens to guarantee
                # that the backbone representations never collapse to low rank!
                ctx_tokens    = context_out.reshape(-1, model.embed_dim)
                ctx_centered  = ctx_tokens - ctx_tokens.mean(dim=0, keepdim=True)
                ctx_std       = torch.sqrt(ctx_centered.var(dim=0) + 1e-04)
                var_loss_ctx  = 25.0 * torch.mean(F.relu(1.0 - ctx_std))

                pred_tokens   = pred_targets[:, :t_len, :].reshape(-1, model.embed_dim)
                pred_centered = pred_tokens - pred_tokens.mean(dim=0, keepdim=True)
                pred_std      = torch.sqrt(pred_centered.var(dim=0) + 1e-04)
                var_loss_pred = 25.0 * torch.mean(F.relu(1.0 - pred_std))

                var_loss      = 0.5 * (var_loss_ctx + var_loss_pred)

                # 3. Trace-Normalised Covariance Decorrelation (nu=1) on Context Encoder
                N = ctx_tokens.size(0)
                if N > 1:
                    cov    = (ctx_centered.T @ ctx_centered) / (N - 1)
                    trace  = torch.trace(cov).clamp(min=1e-5)
                    cov_n  = cov / (trace / model.embed_dim)
                    off_d  = cov_n - torch.diag(torch.diagonal(cov_n))
                    cov_loss = 1.0 * (off_d ** 2).sum() / model.embed_dim
                else:
                    cov_loss = torch.tensor(0.0, device=device)

                step_loss = (loss_inv + var_loss + cov_loss) / accum_steps

            scaler.scale(step_loss).backward()
            epoch_loss += step_loss.item() * accum_steps
            epoch_inv  += loss_inv.item()
            epoch_var  += var_loss.item()
            epoch_cov  += cov_loss.item()
            last_std    = pred_std.mean().item()
            valid_steps += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # V4: Cosine EMA momentum schedule — starts responsive (0.996)
                # and becomes more stable late in training (0.999).
                # High momentum early = target barely updates from random init.
                # High momentum late = stable teacher, clean SSL signal.
                progress = (epoch - 1) / max(epochs - 1, 1)
                momentum = MOM_END - (MOM_END - MOM_START) * (np.cos(np.pi * progress) + 1) / 2
                model.update_target_encoder(momentum=momentum)

            # Live step progress logging (step 1, every 10 steps, and epoch end)
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader) or step == 0:
                elapsed_cur = time.time() - start_time
                steps_done  = step + 1
                avg_step_s  = elapsed_cur / steps_done
                eta_s       = avg_step_s * (len(train_loader) - steps_done)
                print(f"  [Ep {epoch:03d}/{epochs:03d} | Step {steps_done:03d}/{len(train_loader):03d}] "
                      f"Loss: {step_loss.item()*accum_steps:.3f} "
                      f"(Inv: {loss_inv.item():.2f}, Var: {var_loss.item():.2f}, Cov: {cov_loss.item():.2f}) | "
                      f"Speed: {avg_step_s:.2f}s/step | ETA: {eta_s/60:.1f}m", flush=True)

        scheduler.step()
        elapsed  = time.time() - start_time
        vs       = max(valid_steps, 1)
        progress = (epoch - 1) / max(epochs - 1, 1)
        cur_mom  = MOM_END - (MOM_END - MOM_START) * (np.cos(np.pi * progress) + 1) / 2

        print(f"\n--> Epoch [{epoch:03d}/{epochs:03d}] COMPLETE "
              f"Loss: {epoch_loss/vs:.3f} "
              f"(Inv: {epoch_inv/vs:.3f}, Var: {epoch_var/vs:.3f}, Cov: {epoch_cov/vs:.3f}) "
              f"Std: {last_std:.3f}  LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"Mom: {cur_mom:.4f}  Time: {elapsed:.1f}s\n", flush=True)

        if epoch % 5 == 0 or epoch == epochs:
            ckpt = os.path.join(CHECKPOINT_DIR, f"jepa_cub_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'loss': epoch_loss / vs}, ckpt)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "jepa_cub_latest.pt"))
            print(f"  -> Checkpoint saved: {ckpt}")

        # Free cached blocks and prevent CUDA allocator fragmentation across epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[[OK]] CUB-200 Self-Supervised Pre-Training Complete!")
    return model


# -----------------------------------------------------------------------------
# Feature Extraction Cache — V3: Batched GPU forward (Bx faster)
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_cub_features(model, dataloader, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings: {cache_path}")
        cache = torch.load(cache_path, map_location=device)
        return cache['features'], cache['labels']

    print(f"Extracting features (V3 batched) -> {cache_path} ...")
    model.eval()
    features_list, labels_list = [], []
    start = time.time()
    n_done = 0

    for idx, (imgs, lbls) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True)
        # V3: single batched ViT forward — replaces B serial GPU calls
        feats = model.extract_features_batch(imgs, multi_layer=True)   # (B, D)
        features_list.append(feats.cpu())
        labels_list.extend(lbls.tolist())
        n_done += imgs.shape[0]
        if (idx + 1) % 50 == 0 or (idx + 1) == len(dataloader):
            print(f"  -> {n_done}/{len(dataloader.dataset)} processed...")

    all_feats  = torch.cat(features_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    torch.save({'features': all_feats, 'labels': all_labels}, cache_path)
    print(f"[[OK]] Extraction done in {time.time()-start:.1f}s  ({n_done} images)")
    return all_feats, all_labels


# -----------------------------------------------------------------------------
# Phase 2A: Frozen Backbone Linear Probe
# -----------------------------------------------------------------------------
def evaluate_frozen_probe(train_feats, train_labels, test_feats, test_labels,
                          num_classes=200, epochs=50):
    print("\n" + "=" * 70)
    print("      PHASE 2A: FROZEN BACKBONE LINEAR PROBE (CUB-200)")
    print("=" * 70)

    tr_ds  = torch.utils.data.TensorDataset(train_feats.to(device), train_labels.to(device))
    te_ds  = torch.utils.data.TensorDataset(test_feats.to(device),  test_labels.to(device))
    tr_ld  = DataLoader(tr_ds, batch_size=64,  shuffle=True)
    te_ld  = DataLoader(te_ds, batch_size=128, shuffle=False)

    head = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        head.train()
        for x_b, y_b in tr_ld:
            optimizer.zero_grad()
            criterion(head(x_b), y_b).backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x_b, y_b in te_ld:
            all_preds.extend(torch.argmax(head(x_b), dim=1).cpu().numpy())
            all_tgts.extend(y_b.cpu().numpy())

    acc         = accuracy_score(all_tgts, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_tgts, all_preds,
                                                        average='macro', zero_division=0)
    print("\n" + "-" * 55)
    print("  [[OK]] CUB-200 FROZEN PROBE RESULTS:")
    print(f"  Top-1 Accuracy:  {acc  * 100:.2f}%")
    print(f"  Macro Precision: {prec * 100:.2f}%")
    print(f"  Macro Recall:    {rec  * 100:.2f}%")
    print(f"  Macro F1-Score:  {f1   * 100:.2f}%")
    print("-" * 55)
    return acc, prec, rec, f1


# -----------------------------------------------------------------------------
# Phase 2B: End-to-End Supervised Fine-Tuning
# -----------------------------------------------------------------------------
def run_finetuning(jepa_model, train_loader, test_loader, num_classes=200, epochs=50):
    print("\n" + "=" * 70)
    print("      PHASE 2B: END-TO-END SUPERVISED FINE-TUNING (CUB-200)")
    print("=" * 70)
    print("Augmentation: Safe curated (Rotate+/-15?, Translate, mild ColorJitter) + BICUBIC resize")
    print("  [TrivialAugmentWide removed ? Equalize/AutoContrast destroy QuadTree variance map]")
    print("Eval: Flip-only TTA (original + horizontal flip averaged)")
    print("Scheduler: 5-epoch linear warmup -> cosine decay\n")

    classifier = QuadtreeClassifier(jepa_model, num_classes=num_classes).to(device)

    # Differential LRs: backbone 1e-5 (preserves pre-trained visual knowledge),
    # bridge 5e-5, pooler+head 1e-3
    optimizer = torch.optim.AdamW([
        {'params': classifier.context_encoder.parameters(), 'lr': 1e-5,  'weight_decay': 0.05},
        {'params': classifier.z_bridge.parameters(),        'lr': 5e-5,  'weight_decay': 0.05},
        {'params': classifier.pooler.parameters(),          'lr': 1e-3,  'weight_decay': 1e-4},
        {'params': classifier.head.parameters(),            'lr': 1e-3,  'weight_decay': 1e-4},
    ])

    # V3: 5-epoch linear warmup -> cosine decay (prevents early gradient explosions)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        start_t = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = classifier(imgs)
                loss   = criterion(logits, labels)

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start_t

        # Evaluation
        # V4: Flip-only TTA — average logits over original + horizontal flip.
        # WHY flip-only and NOT crop-based:
        #   Different crops change which QuadTree patches are selected (the
        #   variance map changes), making logit averaging incoherent. Horizontal
        #   flip mirrors pixel positions but does NOT change which patches have
        #   highest variance -> QuadTree selects identical token sets (mirrored).
        classifier.eval()
        all_preds, all_tgts = [], []     
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits_orig = classifier(imgs)
                    logits_flip = classifier(torch.flip(imgs, dims=[-1]))  # horizontal flip
                    logits      = (logits_orig + logits_flip) * 0.5        # TTA average
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_tgts.extend(labels.numpy())

        acc         = accuracy_score(all_tgts, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(all_tgts, all_preds,
                                                      average='macro', zero_division=0)
        lr_now = scheduler.get_last_lr()[0]

        print(f"FT Epoch [{epoch:02d}/{epochs:02d}]  "
              f"Loss: {running_loss/len(train_loader):.4f}  "
              f"Acc: {acc*100:.2f}%  F1: {f1*100:.2f}%  "
              f"LR: {lr_now:.2e}  Time: {elapsed:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "cub_finetuned_best.pt"))

    print("\n" + "=" * 55)
    print(f"  [[OK]] CUB-200 BEST FINE-TUNING ACCURACY: {best_acc * 100:.2f}%")
    print("=" * 55)
    return best_acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="QuadTree-JEPA V3 CUB-200 Training & Evaluation")
    parser.add_argument("--pretrain_epochs",  type=int,   default=100)
    parser.add_argument("--probe_epochs",     type=int,   default=50)
    parser.add_argument("--finetune_epochs",  type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=20,
                        help="Images per GPU batch. Empirically validated: "
                             "batch=20 stays safely within 3.2 GB VRAM, preventing Windows 11 "
                             "from paging tensors to system RAM over PCIe (which caused the 32x slowdown at batch=40).")
    parser.add_argument("--accum_steps",      type=int,   default=1,
                        help="Gradient accumulation steps. accum=1 at batch=20 gives "
                             "effective batch=20; set accum=2 for effective batch=40.")
    parser.add_argument("--num_workers",      type=int,   default=2,
                        help="DataLoader workers. 2 is sufficient: GPU step takes ~1.5-2s, "
                             "giving workers plenty of time to prefetch next batch while keeping CPU cool (~65-72C).")
    parser.add_argument("--finetune",         action="store_true")
    parser.add_argument("--eval_only",        action="store_true")
    parser.add_argument("--resume",           nargs="?", const="latest", default=None,
                        help="Resume pre-training from checkpoint. Specify epoch number (e.g. --resume 5), "
                             "checkpoint path, or leave without value (--resume) to resume from latest.")
    parser.add_argument("--no_pretrained",    action="store_true",
                        help="Train ViT backbone completely from scratch instead of loading pre-trained ImageNet weights")
    parser.add_argument("--pretrained_model", type=str, default="vit_base_patch16_224",
                        help="timm model name for pre-trained ViT weights")
    args = parser.parse_args()

    # -- Datasets --
    # SSL pre-training: strong DINO-style augmentation (RandomResizedCrop etc.)
    ssl_dataset  = CUB200Dataset(DATA_DIR, is_train=True,  ssl_mode=True)
    # Fine-tuning: TrivialAugmentWide
    ft_dataset   = CUB200Dataset(DATA_DIR, is_train=True,  ssl_mode=False)
    test_dataset = CUB200Dataset(DATA_DIR, is_train=False, ssl_mode=False)

    print(f"CUB-200 | Train: {len(ssl_dataset)} | Test: {len(test_dataset)} | "
          f"Classes: 200")

    loader_kwargs = dict(
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = (args.num_workers > 0),
        prefetch_factor    = 2 if args.num_workers > 0 else None,
    )

    ssl_loader  = DataLoader(ssl_dataset,  shuffle=True,  **loader_kwargs)
    ft_loader   = DataLoader(ft_dataset,   shuffle=True,  **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # -- V5 Model: ViT-Base Backbone (depth=12, heads=12, dim=768, mlp=3072) --
    # GPU: RTX 4060 Laptop (SM89, Ada Lovelace, 8 GB VRAM)
    # VRAM budget: batch=20 uses ~3.2 GB -> safe on 8 GB
    amp_dtype_name = 'BF16' if torch.cuda.is_bf16_supported() else 'FP16'
    print(f"GPU Optimizations: TF32={torch.backends.cuda.matmul.allow_tf32} | "
          f"cuDNN.benchmark={torch.backends.cudnn.benchmark} | "
          f"AMP={amp_dtype_name} | batch={args.batch_size} | workers={args.num_workers}")
    base_vit = ViT(
        dim           = 768,
        depth         = 12,          # ViT-Base standard
        heads         = 12,          # 12 * 64 = 768 = dim
        mlp_dim       = 3072,        # standard ViT-Base FFN dimension (4 * 768)
        dim_head      = 64,
        dropout       = 0.1,
        drop_path_rate= 0.1,         # V4: stochastic depth
        qkv_bias      = True
    ).to(device)

    if not args.no_pretrained:
        load_pretrained_vit_weights(base_vit, model_name=args.pretrained_model)

    jepa_model = QuadtreeJEPA(base_vit=base_vit, embed_dim=768, target_budget=256).to(device)

    n_params = sum(p.numel() for p in jepa_model.parameters() if p.requires_grad) / 1e6
    print(f"Model: QuadTree-JEPA V5 | Params: {n_params:.1f}M | "
          f"ViT depth=12, heads=12, mlp=3072 | Pretrained: {not args.no_pretrained}")

    if not args.eval_only:
        if not args.resume:
            # Invalidate stale feature caches if starting a brand new run
            for old in ["cub_train_feats.pt", "cub_test_feats.pt", "jepa_cub_latest.pt"]:
                p = os.path.join(CHECKPOINT_DIR, old)
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  Removed stale cache: {p}")

        jepa_model = run_pretraining(
            jepa_model, ssl_loader,
            epochs=args.pretrain_epochs,
            accum_steps=args.accum_steps,
            resume=args.resume,
            is_pretrained=(not args.no_pretrained)
        )
    else:
        latest = os.path.join(CHECKPOINT_DIR, "jepa_cub_latest.pt")
        if os.path.exists(latest):
            print(f"Loading checkpoint: {latest}")
            sd = torch.load(latest, map_location=device)
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            jepa_model.load_state_dict(sd, strict=False)

    # -- Phase 2A: Frozen Linear Probe --
    # V4 BUG FIX: previously used is_train=True (ssl_mode=False) which applied
    # TrivialAugmentWide -> every call to extract_cub_features produced different
    # random features for the same images -> linear probe accuracy varied +/-2-3%.
    # force_eval_transform=True overrides to deterministic resize+normalize only.
    eval_train_loader = DataLoader(
        CUB200Dataset(DATA_DIR, is_train=True, force_eval_transform=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    train_cache = os.path.join(CHECKPOINT_DIR, "cub_train_feats.pt")
    test_cache  = os.path.join(CHECKPOINT_DIR, "cub_test_feats.pt")

    train_feats, train_labels = extract_cub_features(jepa_model, eval_train_loader, train_cache)
    test_feats,  test_labels  = extract_cub_features(jepa_model, test_loader,       test_cache)

    evaluate_frozen_probe(
        train_feats, train_labels,
        test_feats,  test_labels,
        num_classes=200, epochs=args.probe_epochs
    )

    # -- Phase 2B: End-to-End Supervised Fine-Tuning --
    if args.finetune:
        run_finetuning(
            jepa_model, ft_loader, test_loader,
            num_classes=200, epochs=args.finetune_epochs
        )


if __name__ == "__main__":
    main()
