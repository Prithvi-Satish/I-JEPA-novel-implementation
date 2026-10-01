import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from vit_pytorch.vit import ViT, Transformer
from quadtree_jepa import QuadtreeJEPA, QuadtreeTokenizer, ZAxisFusionBridge
from train_and_evaluate_jepa import LabeledPlantDataset, DATA_DIR, TARGET_SIZE, EMBED_DIM, MAX_SEQ_LEN

# ==========================================
# FAST BATCHED FINE-TUNING PIPELINE
# ==========================================
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

class FastBatchedClassifier(nn.Module):
    def __init__(self, jepa_model, num_classes=18):
        super().__init__()
        self.tokenizer = jepa_model.tokenizer
        self.z_bridge = jepa_model.z_bridge
        self.context_encoder = jepa_model.context_encoder
        self.max_seq_len = jepa_model.max_seq_len
        
        # Attentive Deep Classification Head
        self.attn_pool = nn.MultiheadAttention(EMBED_DIM, num_heads=4, batch_first=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def extract_image_tokens(self, img):
        if img.dim() == 4:
            img = img.squeeze(0)
        patches, metadata = self.tokenizer(img)
        if not metadata:
            device = next(self.parameters()).device
            return torch.zeros(1, EMBED_DIM, device=device)
        tokens = self.z_bridge(patches, metadata)
        if len(tokens) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, EMBED_DIM, device=device)
        seq_len = min(len(tokens), self.max_seq_len)
        return tokens[:seq_len]

    def forward_tokens(self, batched_tokens, lengths):
        # batched_tokens: (B, max_L, D)
        B = batched_tokens.shape[0]
        context_out = self.context_encoder(batched_tokens)
        
        # Attentive pooling with learnable query token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        pooled, _ = self.attn_pool(cls_tokens, context_out, context_out)
        pooled = pooled.squeeze(1)
        return self.head(pooled)

def collate_quadtree_batch(batch, model, device):
    images, labels = batch
    token_list = []
    for img in images:
        tokens = model.extract_image_tokens(img.to(device))
        token_list.append(tokens)
        
    max_len = max(t.shape[0] for t in token_list)
    B = len(token_list)
    D = token_list[0].shape[1]
    
    padded = torch.zeros(B, max_len, D, device=device)
    lengths = []
    for i, t in enumerate(token_list):
        L = t.shape[0]
        padded[i, :L] = t
        lengths.append(L)
        
    labels_tensor = labels.to(device)
    return padded, lengths, labels_tensor

def main():
    print("=" * 70)
    print("  QUADTREE-JEPA FAST BATCHED END-TO-END FINE-TUNING PIPELINE")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES)
    
    # 1. Initialize Base QuadTree-JEPA Architecture
    base_vit = ViT(
        dim=EMBED_DIM,
        depth=6,
        heads=8,
        mlp_dim=1536,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    jepa_model = QuadtreeJEPA(base_vit=base_vit, embed_dim=EMBED_DIM, max_seq_len=MAX_SEQ_LEN).to(device)
    
    # 2. Load Checkpoint
    ckpt_path = "./checkpoints/jepa_plant_epoch15.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print(f"[*] Loading pretrained JEPA weights from: {ckpt_path}")
    jepa_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    print("[*] Successfully loaded all backbone weights!\n")
    
    # 3. Create Fast Batched Classifier
    classifier = FastBatchedClassifier(jepa_model, num_classes=num_classes).to(device)
    
    # 4. Prepare Datasets
    train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE, is_train=True)
    test_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "test"), target_size=TARGET_SIZE, is_train=False)
    print(f"[*] Training samples: {len(train_dataset):,} | Testing samples: {len(test_dataset):,}\n")
    
    # Hyperparameters
    BATCH_SIZE = 16
    GRAD_ACCUM = 2
    EPOCHS = 10
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    param_groups = [
        {"params": classifier.context_encoder.parameters(), "lr": 3e-5, "weight_decay": 0.05},
        {"params": classifier.z_bridge.parameters(), "lr": 3e-5, "weight_decay": 0.05},
        {"params": classifier.attn_pool.parameters(), "lr": 5e-4, "weight_decay": 1e-3},
        {"params": classifier.head.parameters(), "lr": 1e-3, "weight_decay": 1e-3}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    print("-" * 70)
    print(f"[*] Launching Fast Batched Fine-Tuning ({EPOCHS} Epochs, Batch Size: {BATCH_SIZE})...")
    print("-" * 70)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            padded_tokens, lengths, targets = collate_quadtree_batch(batch, classifier, device)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = classifier.forward_tokens(padded_tokens, lengths)
                loss = criterion(logits, targets) / GRAD_ACCUM
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM
            
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            if (step + 1) % 200 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - epoch_start
                print(f"  -> Epoch [{epoch:02d}/{EPOCHS}] | Step [{step+1:>4}/{len(train_loader)}] | Train Acc: {(correct/total)*100:5.2f}% | Loss: {running_loss/(step+1):.4f} | Time: {elapsed:.1f}s", flush=True)
                
        scheduler.step()
        epoch_dur = (time.time() - epoch_start) / 60
        train_acc = (correct / total) * 100
        print(f"\n[*] Epoch [{epoch:02d}/{EPOCHS}] Complete in {epoch_dur:.1f}m | Train Acc: {train_acc:.2f}% | Head LR: {optimizer.param_groups[3]['lr']:.2e}\n", flush=True)
        
    # Final Benchmark Evaluation on Test Set
    print("=" * 70)
    print(f"[*] Running Final Fine-Tuned Benchmark Evaluation on {len(test_dataset):,} Test Images...")
    print("=" * 70)
    
    classifier.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            padded_tokens, lengths, targets = collate_quadtree_batch(batch, classifier, device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = classifier.forward_tokens(padded_tokens, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    test_preds = np.array(all_preds)
    test_targets = np.array(all_targets)
    
    final_acc = accuracy_score(test_targets, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
    
    print("\n" + "=" * 70)
    print(f"  FINE-TUNED BENCHMARK ACCURACY:  {final_acc * 100:.2f}% (Top-1)")
    print(f"  FINE-TUNED MACRO F1-SCORE:      {f1 * 100:.2f}%")
    print(f"  FINE-TUNED MACRO PRECISION:     {prec * 100:.2f}%")
    print(f"  FINE-TUNED MACRO RECALL:        {rec * 100:.2f}%")
    print("=" * 70)
    
    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4, zero_division=0))
    
    # Save Final Fine-Tuned Checkpoint
    out_ckpt = "./checkpoints/jepa_plant_finetuned.pt"
    torch.save(classifier.state_dict(), out_ckpt)
    print(f"[*] Fine-tuned model checkpoint saved to: {os.path.abspath(out_ckpt)}")
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(13, 11))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Quadtree-JEPA Fine-Tuned Model (Acc: {final_acc*100:.2f}%)", fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=35, ha="right", fontsize=8)
    plt.yticks(tick_marks, CLASSES, fontsize=8)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         horizontalalignment="center",
                         color="white" if val > thresh else "black",
                         fontsize=7, fontweight="bold")
                         
    plt.ylabel('Ground Truth', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    cm_path = "confusion_matrix_finetuned.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix plot saved to: {os.path.abspath(cm_path)}")

if __name__ == "__main__":
    main()
