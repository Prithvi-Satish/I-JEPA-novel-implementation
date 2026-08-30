import os
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from vit_pytorch.vit import ViT
from quadtree_jepa import QuadtreeJEPA, QuadtreeClassifier

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
PRETRAIN_EPOCHS = 75
PROBE_EPOCHS = 40
FINETUNE_EPOCHS = 25
BATCH_SIZE = 1  # Processed per image due to dynamic quadtree sequence lengths
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END = 0.9999
EMBED_DIM = 768
MAX_SEQ_LEN = 800
TARGET_SIZE = 504
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data/plant_dataset"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATASET DEFINITION
# ==========================================
class LabeledPlantDataset(Dataset):
    def __init__(self, root_dir, target_size=504):
        self.root_dir = root_dir
        self.target_size = target_size
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if any(fname.endswith(ext) for ext in valid_exts):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
                    
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
# PHASE 1: SELF-SUPERVISED PRE-TRAINING (MOD-04)
# ==========================================
def run_pretraining(model, train_loader, optimizer, scheduler, monitor):
    print("=" * 65)
    print(f"  PHASE 1: SELF-SUPERVISED QUADTREE-JEPA PRE-TRAINING ({PRETRAIN_EPOCHS} EPOCHS)")
    print("=" * 65)
    
    start_time = time.time()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        model.target_encoder.eval()
        
        optimizer.zero_grad()
        running_loss = 0.0
        valid_steps = 0
        
        alpha = EMA_MOMENTUM_START + (EMA_MOMENTUM_END - EMA_MOMENTUM_START) * (epoch / PRETRAIN_EPOCHS)
        
        last_valid_pred = None
        last_valid_true = None
        last_valid_t_len = 0
        
        accum_preds = []
        accum_align_losses = []
        
        epoch_start = time.time()
        for step, (images, _) in enumerate(train_loader):
            img = images[0].to(device)  # single image processing
            pred_targets, true_targets, target_mask, t_len = model(img)
            
            if pred_targets is None or t_len == 0:
                continue
                
            last_valid_pred = pred_targets
            last_valid_true = true_targets
            last_valid_t_len = t_len
                
            pred_active = pred_targets[0, :t_len, :]
            true_active = true_targets[0, :t_len, :]
            
            # Unit norm alignment loss for this image
            pred_norm = F.normalize(pred_active, p=2, dim=-1)
            true_norm = F.normalize(true_active, p=2, dim=-1)
            align_loss = F.mse_loss(pred_norm, true_norm, reduction='sum') / (t_len * EMBED_DIM)
            
            accum_preds.append(pred_active)
            accum_align_losses.append(align_loss)
            
            # Batch-level variance regularization across accumulated gradient steps
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                if len(accum_preds) > 0:
                    all_pred_tokens = torch.cat(accum_preds, dim=0)
                    pred_std = torch.sqrt(all_pred_tokens.var(dim=0) + 1e-04)
                    variance_loss = torch.mean(F.relu(1.0 - pred_std))
                    
                    mean_align_loss = torch.stack(accum_align_losses).mean()
                    total_loss = (mean_align_loss * 10.0) + variance_loss
                    
                    total_loss.backward()
                    
                    nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(model.z_bridge.parameters(), max_norm=1.0)
                    
                    optimizer.step()
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
            print(f"Epoch [{epoch:02d}/{PRETRAIN_EPOCHS}] ({epoch_duration:.1f}s) | Loss: {avg_loss:.5f} | Latent Std: {std_dev:.4f} | CosSim: {cos_sim:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"Epoch [{epoch:02d}/{PRETRAIN_EPOCHS}] ({epoch_duration:.1f}s) | Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    total_time = (time.time() - start_time) / 60
    print(f"\n Pre-training completed in {total_time:.2f} minutes.")
    
    # Save pretrained weights
    ckpt_path = os.path.join(CHECKPOINT_DIR, "jepa_plant_75epochs.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f" Checkpoint saved to: {ckpt_path}\n")

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
            img = img.to(device)
            feat = model.extract_features(img)  # Shape: (1, 768)
            embeddings.append(feat.cpu().squeeze(0))
            labels.append(label)
    return torch.stack(embeddings), torch.tensor(labels)

def train_linear_probe(train_feats, train_labels, num_classes=3, epochs=40):
    print("=" * 65)
    print("  PHASE 2A: TRAINING LINEAR CLASSIFIER (PROBING FROZEN JEPA FEATURES)")
    print("=" * 65)
    
    linear_head = nn.Linear(EMBED_DIM, num_classes).to(device)
    probe_optimizer = torch.optim.Adam(linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
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
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Probe Epoch [{epoch:02d}/{epochs}] | Cross-Entropy Loss: {running_loss/len(loader):.4f}")
            
    return linear_head

# ==========================================
# PHASE 2B: END-TO-END DISCRIMINATIVE FINE-TUNING (MOD-02)
# ==========================================
def train_finetuned_classifier(base_jepa_model, train_dataset, num_classes=3, epochs=25):
    print("=" * 65)
    print(f"  PHASE 2B: END-TO-END DISCRIMINATIVE FINE-TUNING ({epochs} EPOCHS)")
    print("=" * 65)
    
    classifier = QuadtreeClassifier(base_jepa_model, num_classes=num_classes).to(device)
    
    # Discriminative parameter groups
    # Backbone: 1e-5 (gentle adaptation, preserves SSL priors)
    # Head: 1e-3 (rapid convergence on class decision boundaries)
    param_groups = [
        {"params": classifier.context_encoder.parameters(), "lr": 1e-5, "weight_decay": 0.05},
        {"params": classifier.z_bridge.parameters(), "lr": 1e-5, "weight_decay": 0.05},
        {"params": classifier.head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, (images, labels) in enumerate(loader):
            img = images[0].to(device)
            label = labels.to(device)
            
            logits = classifier(img)
            loss = criterion(logits, label) / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Fine-Tune Epoch [{epoch:02d}/{epochs}] | Cross-Entropy Loss: {running_loss/len(loader):.4f} | Head LR: {optimizer.param_groups[2]['lr']:.2e} | Backbone LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    return classifier

# ==========================================
# PHASE 3: EVALUATION & METRIC CALCULATION
# ==========================================
def evaluate_model(linear_head, test_feats, test_labels, class_names):
    print("\n" + "=" * 65)
    print("  PHASE 3A: BENCHMARK EVALUATION (FROZEN LINEAR PROBE - 60 IMAGES)")
    print("=" * 65)
    
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
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Quadtree-JEPA Frozen Linear Probe Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right")
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")
                     
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = "confusion_matrix_jepa.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Confusion matrix plot saved to: {os.path.abspath(plot_path)}")
    return acc, f1

def evaluate_finetuned_model(classifier, test_dataset, class_names):
    print("\n" + "=" * 65)
    print("  PHASE 3B: BENCHMARK EVALUATION (FINE-TUNED CLASSIFIER - 60 IMAGES)")
    print("=" * 65)
    
    classifier.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            img = img.to(device)
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
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Quadtree-JEPA Fine-Tuned Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right")
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")
                     
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
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load Data
    train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE)
    test_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "test"), target_size=TARGET_SIZE)
    
    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)} | Testing samples: {len(test_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
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
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6)
    monitor = TrainingMonitor()
    
    # 3. Phase 1: Pre-training (75 Epochs)
    run_pretraining(model, train_loader, optimizer, scheduler, monitor)
    
    # 4. Phase 2A: Feature Extraction & Linear Probe Training (Frozen Backbone)
    train_feats, train_labels = extract_dataset_embeddings(model, train_dataset)
    test_feats, test_labels = extract_dataset_embeddings(model, test_dataset)
    
    linear_head = train_linear_probe(train_feats, train_labels, num_classes=len(train_dataset.classes), epochs=PROBE_EPOCHS)
    
    # 5. Phase 2B: End-to-End Discriminative Fine-Tuning (Unfrozen Backbone)
    ft_classifier = train_finetuned_classifier(model, train_dataset, num_classes=len(train_dataset.classes), epochs=FINETUNE_EPOCHS)
    
    # 6. Phase 3: Benchmark Evaluations & Comparison
    evaluate_model(linear_head, test_feats, test_labels, train_dataset.classes)
    evaluate_finetuned_model(ft_classifier, test_dataset, train_dataset.classes)

if __name__ == "__main__":
    main()
