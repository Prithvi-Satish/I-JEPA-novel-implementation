import os
import sys
import time
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from vit_pytorch.vit import ViT
from quadtree_jepa import QuadtreeJEPA, QuadtreeClassifier

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
PRETRAIN_EPOCHS = 30
PROBE_EPOCHS = 30
FINETUNE_EPOCHS = 20
BATCH_SIZE = 1  # Dynamic quadtree sequence lengths processed per image
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END = 0.9999
EMBED_DIM = 768
MAX_SEQ_LEN = 800
TARGET_SIZE = 504
NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 5  # Save checkpoint every 5 epochs
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data/plant_dataset"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

# ==========================================
# DATASET DEFINITION
# ==========================================
class LabeledPlantDataset(Dataset):
    def __init__(self, root_dir, target_size=504, is_train=True):
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_train = is_train
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if any(fname.endswith(ext) for ext in valid_exts):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
                    
        if is_train:
            self.transform = T.Compose([
                T.Resize((self.target_size, self.target_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.target_size, self.target_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert('RGB')
            tensor = self.transform(rgb)
        return tensor, label

# ==========================================
# LATENT HEALTH MONITOR
# ==========================================
class TrainingMonitor:
    def __init__(self, collapse_threshold=0.01):
        self.collapse_threshold = collapse_threshold

    @torch.no_grad()
    def check(self, student_latents, target_latents, t_len, step, epoch):
        valid_student = student_latents[0, :t_len, :]
        valid_target = target_latents[0, :t_len, :]
        std_dev = valid_student.std(dim=0).mean().item()
        cos_sim = F.cosine_similarity(
            valid_student.mean(dim=0, keepdim=True),
            valid_target.mean(dim=0, keepdim=True),
            dim=-1
        ).mean().item()
        
        if std_dev < self.collapse_threshold:
            raise RuntimeError(f"[CRITICAL] Representation collapse detected at Epoch {epoch}! Std Dev: {std_dev:.5f}")
        return std_dev, cos_sim

# ==========================================
# PHASE 1: SELF-SUPERVISED PRE-TRAINING (HIGH PERFORMANCE + AMP)
# ==========================================
def run_pretraining(model, train_loader, optimizer, scheduler, monitor, resume_path=None, total_epochs=PRETRAIN_EPOCHS):
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"\n[*] Resuming QuadTree-JEPA pre-training from: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        base_name = os.path.basename(resume_path)
        digits = ''.join(filter(str.isdigit, base_name))
        if digits:
            start_epoch = int(digits) + 1
        print(f"[*] Resuming from Epoch {start_epoch:02d} up to {total_epochs:02d}...\n")
        for _ in range(1, start_epoch):
            scheduler.step()

    print("=" * 70)
    print(f"  PHASE 1: SELF-SUPERVISED QUADTREE-JEPA PRE-TRAINING ({total_epochs} EPOCHS)")
    print(f"  Acceleration: Mixed Precision (AMP FP16: {use_amp}) | Workers: {NUM_WORKERS}")
    print("  Loss Regularization: Variance (VICReg) + Covariance Decorrelation (Orthogonal Dimensions)")
    print("=" * 70)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        model.target_encoder.eval()
        
        optimizer.zero_grad()
        running_loss = 0.0
        valid_steps = 0
        
        alpha = EMA_MOMENTUM_START + (EMA_MOMENTUM_END - EMA_MOMENTUM_START) * (epoch / total_epochs)
        
        last_valid_pred = None
        last_valid_true = None
        last_valid_t_len = 0
        
        accum_preds = []
        accum_align_losses = []
        
        epoch_start = time.time()
        for step, (images, _) in enumerate(train_loader):
            img = images[0].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_targets, true_targets, _, t_len = model(img)
                
                if pred_targets is None or t_len == 0:
                    continue
                    
                last_valid_pred = pred_targets.detach()
                last_valid_true = true_targets.detach()
                last_valid_t_len = t_len
                    
                pred_active = pred_targets[0, :t_len, :]
                true_active = true_targets[0, :t_len, :]
                
                # Unit norm alignment loss
                pred_norm = F.normalize(pred_active, p=2, dim=-1)
                true_norm = F.normalize(true_active, p=2, dim=-1)
                align_loss = F.mse_loss(pred_norm, true_norm, reduction='sum') / (t_len * EMBED_DIM)
                
                accum_preds.append(pred_active)
                accum_align_losses.append(align_loss)
            
            # Batch-level variance + covariance decorrelation regularization across accumulated steps
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                if len(accum_preds) > 0:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        all_pred_tokens = torch.cat(accum_preds, dim=0)
                        
                        # 1. Variance Regularization (Forces dimension std >= 1.0)
                        pred_centered = all_pred_tokens - all_pred_tokens.mean(dim=0, keepdim=True)
                        pred_std = torch.sqrt(pred_centered.var(dim=0) + 1e-04)
                        variance_loss = torch.mean(F.relu(1.0 - pred_std))
                        
                        # 2. Covariance Decorrelation Loss (Forces dimensions to be orthogonal/uncorrelated)
                        N = all_pred_tokens.size(0)
                        cov_matrix = (pred_centered.T @ pred_centered) / (N - 1)
                        off_diagonal = cov_matrix - torch.diag(torch.diagonal(cov_matrix))
                        cov_loss = (off_diagonal ** 2).sum() / EMBED_DIM
                        
                        mean_align_loss = torch.stack(accum_align_losses).mean()
                        total_loss = (mean_align_loss * 10.0) + variance_loss + (cov_loss * 0.04)
                    
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    
                    nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(model.z_bridge.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    model.update_target_encoder(momentum=alpha)
                    
                    running_loss += total_loss.item()
                    valid_steps += 1
                    
                    accum_preds.clear()
                    accum_align_losses.clear()
                
        scheduler.step()
        epoch_duration = time.time() - epoch_start
        
        avg_loss = running_loss / max(valid_steps, 1)
        if last_valid_pred is not None:
            std_dev, cos_sim = monitor.check(last_valid_pred, last_valid_true, last_valid_t_len, step, epoch)
            print(f"Epoch [{epoch:02d}/{PRETRAIN_EPOCHS}] ({epoch_duration/60:.1f}m) | Loss: {avg_loss:.5f} | Latent Std: {std_dev:.4f} | CosSim: {cos_sim:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Epoch [{epoch:02d}/{PRETRAIN_EPOCHS}] ({epoch_duration/60:.1f}m) | Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
        # Periodic checkpoint save
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == PRETRAIN_EPOCHS:
            ckpt_name = f"jepa_plant_epoch{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, ckpt_name))
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "jepa_plant_latest.pt"))
            print(f"  -> Checkpoint successfully saved: {ckpt_name}")
            
    total_time = (time.time() - start_time) / 60
    print(f"\n Pre-training completed in {total_time:.2f} minutes.")
    
    final_path = os.path.join(CHECKPOINT_DIR, f"jepa_plant_{PRETRAIN_EPOCHS}epochs.pt")
    torch.save(model.state_dict(), final_path)
    print(f" Final checkpoint saved to: {final_path}\n")

# ==========================================
# PHASE 2A: LINEAR PROBING & FEATURE EXTRACTION
# ==========================================
def extract_dataset_embeddings(model, dataset):
    embeddings = []
    labels = []
    print(f"Extracting frozen latent features for {len(dataset)} images (Multi-Scale)...")
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                feat = model.extract_features(img)
            embeddings.append(feat.float().cpu().squeeze(0))
            labels.append(label)
            if (i + 1) % 1000 == 0 or (i + 1) == len(dataset):
                print(f"  -> Extracted {i+1}/{len(dataset)} features...")
    return torch.stack(embeddings), torch.tensor(labels)

def train_linear_probe(train_feats, train_labels, num_classes=18, epochs=PROBE_EPOCHS):
    print("=" * 70)
    print(f"  PHASE 2A: TRAINING LINEAR CLASSIFIER (PROBING FROZEN JEPA FEATURES, {epochs} EPOCHS)")
    print("=" * 70)
    
    linear_head = nn.Linear(EMBED_DIM, num_classes).to(device)
    probe_optimizer = torch.optim.Adam(linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        linear_head.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probe_optimizer.zero_grad()
            logits = linear_head(x)
            loss = criterion(logits, y)
            loss.backward()
            probe_optimizer.step()
            running_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Probe Epoch [{epoch:02d}/{epochs}] | Cross-Entropy Loss: {running_loss/len(loader):.4f}")
            
    return linear_head

# ==========================================
# PHASE 2B: END-TO-END DISCRIMINATIVE FINE-TUNING
# ==========================================
def train_finetuned_classifier(base_jepa_model, train_dataset, num_classes=18, epochs=FINETUNE_EPOCHS):
    print("=" * 70)
    print(f"  PHASE 2B: END-TO-END DISCRIMINATIVE FINE-TUNING ({epochs} EPOCHS)")
    print("=" * 70)
    
    classifier = QuadtreeClassifier(base_jepa_model, num_classes=num_classes).to(device)
    
    param_groups = [
        {"params": classifier.context_encoder.parameters(), "lr": 1e-5, "weight_decay": 0.05},
        {"params": classifier.z_bridge.parameters(), "lr": 1e-5, "weight_decay": 0.05},
        {"params": classifier.head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, (images, labels) in enumerate(loader):
            img = images[0].to(device, non_blocking=True)
            label = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = classifier(img)
                loss = criterion(logits, label) / GRAD_ACCUM_STEPS
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Fine-Tune Epoch [{epoch:02d}/{epochs}] | Cross-Entropy Loss: {running_loss/len(loader):.4f} | Head LR: {optimizer.param_groups[2]['lr']:.2e} | Backbone LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    return classifier

# ==========================================
# PHASE 3: EVALUATION & METRIC CALCULATION
# ==========================================
def evaluate_model(linear_head, test_feats, test_labels, class_names):
    print("\n" + "=" * 70)
    print(f"  PHASE 3A: BENCHMARK EVALUATION (FROZEN LINEAR PROBE - {len(test_labels)} IMAGES)")
    print("=" * 70)
    
    linear_head.eval()
    with torch.no_grad():
        x = test_feats.to(device)
        logits = linear_head(x)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        targets = test_labels.numpy()
        
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    cm = confusion_matrix(targets, preds)
    
    print("\n" + "-" * 40)
    print(f"  Frozen Probe Accuracy:        {acc * 100:.2f}%")
    print(f"  Frozen Probe Macro Precision: {precision * 100:.2f}%")
    print(f"  Frozen Probe Macro Recall:    {recall * 100:.2f}%")
    print(f"  Frozen Probe Macro F1-Score:  {f1 * 100:.2f}%")
    print("-" * 40)
    
    print("\nDetailed Per-Class Classification Report (Frozen Probe):")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Quadtree-JEPA Frozen Linear Probe Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right", fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         horizontalalignment="center",
                         color="white" if val > thresh else "black",
                         fontsize=7, fontweight="bold")
                         
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = "confusion_matrix_jepa.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Confusion matrix plot saved to: {os.path.abspath(plot_path)}")
    return acc, f1

def evaluate_finetuned_model(classifier, test_dataset, class_names):
    print("\n" + "=" * 70)
    print(f"  PHASE 3B: BENCHMARK EVALUATION (FINE-TUNED CLASSIFIER - {len(test_dataset)} IMAGES)")
    print("=" * 70)
    
    classifier.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = classifier(img)
            pred = torch.argmax(logits, dim=-1).item()
            all_preds.append(pred)
            all_targets.append(label)
            
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    cm = confusion_matrix(targets, preds)
    
    print("\n" + "-" * 40)
    print(f"  Fine-Tuned Accuracy:        {acc * 100:.2f}%")
    print(f"  Fine-Tuned Macro Precision: {precision * 100:.2f}%")
    print(f"  Fine-Tuned Macro Recall:    {recall * 100:.2f}%")
    print(f"  Fine-Tuned Macro F1-Score:  {f1 * 100:.2f}%")
    print("-" * 40)
    
    print("\nDetailed Per-Class Classification Report (Fine-Tuned):")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Quadtree-JEPA Fine-Tuned Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right", fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         horizontalalignment="center",
                         color="white" if val > thresh else "black",
                         fontsize=7, fontweight="bold")
                         
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = "confusion_matrix_finetuned.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Fine-tuned confusion matrix plot saved to: {os.path.abspath(plot_path)}")
    return acc, f1
# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="QuadTree-JEPA Training and Dual Evaluation Pipeline")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Skip pretraining and run linear probe + fine-tuning directly")
    parser.add_argument("--pretrain_epochs", type=int, default=PRETRAIN_EPOCHS, help="Total pretrain epochs")
    parser.add_argument("--probe_epochs", type=int, default=PROBE_EPOCHS, help="Probe epochs")
    parser.add_argument("--finetune_epochs", type=int, default=FINETUNE_EPOCHS, help="Fine-tuning epochs")
    args = parser.parse_args()

    print("=" * 70)
    print("  QUADTREE-JEPA END-TO-END BENCHMARK PIPELINE")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    # 1. Prepare Datasets & Loaders
    train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE, is_train=True)
    eval_train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE, is_train=False)
    test_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "test"), target_size=TARGET_SIZE, is_train=False)
    num_classes = len(train_dataset.classes)
    
    print(f"Classes ({num_classes}): {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset):,} | Testing samples: {len(test_dataset):,}\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=torch.cuda.is_available()
    )
    
    # 2. Build Model
    base_vit = ViT(
        dim=EMBED_DIM,
        depth=6,
        heads=8,
        mlp_dim=1536,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    
    model = QuadtreeJEPA(base_vit=base_vit, embed_dim=EMBED_DIM, max_seq_len=MAX_SEQ_LEN).to(device)
    
    param_groups = [
        {"params": model.context_encoder.parameters()},
        {"params": model.predictor.parameters()},
        {"params": model.z_bridge.parameters()}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs, eta_min=1e-6)
    monitor = TrainingMonitor()
    
    # 3. Phase 1: Pre-training (or load checkpoint if --eval_only)
    if args.eval_only:
        ckpt = args.resume or os.path.join(CHECKPOINT_DIR, "jepa_plant_latest.pt")
        if not os.path.exists(ckpt) and os.path.exists(os.path.join(CHECKPOINT_DIR, "jepa_plant_epoch15.pt")):
            ckpt = os.path.join(CHECKPOINT_DIR, "jepa_plant_epoch15.pt")
        print(f"\n[*] --eval_only mode active. Loading checkpoint: {ckpt}")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"[*] Successfully loaded weights from {ckpt}\n")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    else:
        run_pretraining(model, train_loader, optimizer, scheduler, monitor, resume_path=args.resume, total_epochs=args.pretrain_epochs)
    
    # 4. Phase 2A: Feature Extraction & Linear Probe Training (Frozen Backbone)
    train_feats, train_labels = extract_dataset_embeddings(model, eval_train_dataset)
    test_feats, test_labels = extract_dataset_embeddings(model, test_dataset)
    
    linear_head = train_linear_probe(train_feats, train_labels, num_classes=num_classes, epochs=args.probe_epochs)
    
    # 5. Phase 2B: End-to-End Discriminative Fine-Tuning (Unfrozen Backbone)
    ft_classifier = train_finetuned_classifier(model, train_dataset, num_classes=num_classes, epochs=args.finetune_epochs)
    
    # 6. Phase 3: Benchmark Evaluations & Comparison
    evaluate_model(linear_head, test_feats, test_labels, train_dataset.classes)
    evaluate_finetuned_model(ft_classifier, test_dataset, train_dataset.classes)

if __name__ == "__main__":
    main()
