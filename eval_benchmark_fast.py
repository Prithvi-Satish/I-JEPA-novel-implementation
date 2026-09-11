import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ==========================================
# FAST BENCHMARK EVALUATOR (CACHED EMBEDDINGS)
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

def main():
    print("=" * 70)
    print("  QUADTREE-JEPA HIGH-PERFORMANCE BENCHMARK EVALUATOR")
    print("=" * 70)
    
    train_cache_path = "./checkpoints/train_embeddings_cache.pt"
    test_cache_path = "./checkpoints/test_embeddings_cache.pt"
    
    if not os.path.exists(train_cache_path) or not os.path.exists(test_cache_path):
        raise FileNotFoundError("Cached embeddings not found! Ensure train_embeddings_cache.pt exists.")
        
    print(f"[*] Loading cached multi-scale latent features from disk...")
    train_data = torch.load(train_cache_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_cache_path, map_location='cpu', weights_only=False)
    
    train_feats, train_labels = train_data['feats'], train_data['labels']
    test_feats, test_labels = test_data['feats'], test_data['labels']
    
    print(f"[*] Train set: {train_feats.shape[0]:,} samples (Dim: {train_feats.shape[1]})")
    print(f"[*] Test set:  {test_feats.shape[0]:,} samples (Dim: {test_feats.shape[1]})\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASSES)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_feats, train_labels),
        batch_size=128, shuffle=True
    )
    
    # -------------------------------------------------------------
    # MODEL 1: FROZEN LINEAR PROBE (Standard SSL Benchmark)
    # -------------------------------------------------------------
    print("-" * 70)
    print(" [1/2] Training Standard Frozen Linear Probe (50 Epochs)...")
    print("-" * 70)
    linear_model = nn.Linear(768, num_classes).to(device)
    linear_opt = torch.optim.AdamW(linear_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        linear_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            linear_opt.zero_grad()
            out = linear_model(x)
            loss = criterion(out, y)
            loss.backward()
            linear_opt.step()
            
    linear_model.eval()
    with torch.no_grad():
        preds_lin = linear_model(test_feats.to(device)).argmax(dim=-1).cpu().numpy()
        targets = test_labels.numpy()
        
    acc_lin = accuracy_score(targets, preds_lin)
    _, _, f1_lin, _ = precision_recall_fscore_support(targets, preds_lin, average='macro', zero_division=0)
    print(f" -> Frozen Linear Probe:  Accuracy = {acc_lin * 100:.2f}% | Macro F1 = {f1_lin * 100:.2f}%\n")
    
    # -------------------------------------------------------------
    # MODEL 2: DEEP RESIDUAL MULTI-SCALE CLASSIFIER
    # -------------------------------------------------------------
    print("-" * 70)
    print(" [2/2] Training Deep Non-Linear Residual Classifier (100 Epochs)...")
    print("-" * 70)
    
    class DeepResidualClassifier(nn.Module):
        def __init__(self, in_dim=768, hidden_dim=1024, num_classes=18):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            )
            self.res1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.head = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            h = self.input_layer(x)
            h = h + self.res1(h)
            return self.head(h)
            
    deep_model = DeepResidualClassifier(num_classes=num_classes).to(device)
    deep_opt = torch.optim.AdamW(deep_model.parameters(), lr=8e-4, weight_decay=1e-3)
    deep_sched = torch.optim.lr_scheduler.CosineAnnealingLR(deep_opt, T_max=100)
    crit_smooth = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    for epoch in range(100):
        deep_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            deep_opt.zero_grad()
            out = deep_model(x)
            loss = crit_smooth(out, y)
            loss.backward()
            deep_opt.step()
        deep_sched.step()
        
    deep_model.eval()
    with torch.no_grad():
        preds_deep = deep_model(test_feats.to(device)).argmax(dim=-1).cpu().numpy()
        
    acc_deep = accuracy_score(targets, preds_deep)
    prec_deep, rec_deep, f1_deep, _ = precision_recall_fscore_support(targets, preds_deep, average='macro', zero_division=0)
    print(f" -> Deep Residual Probe: Accuracy = {acc_deep * 100:.2f}% | Macro F1 = {f1_deep * 100:.2f}%\n")
    
    # -------------------------------------------------------------
    # FINAL BENCHMARK SUMMARY & REPORT
    # -------------------------------------------------------------
    print("=" * 70)
    print("                      FINAL BENCHMARK COMPARISON                      ")
    print("=" * 70)
    print(f"{'Evaluation Method':<30} | {'Top-1 Accuracy':<15} | {'Macro F1-Score':<15}")
    print("-" * 70)
    print(f"{'Frozen Linear Probe':<30} | {acc_lin*100:>13.2f}% | {f1_lin*100:>13.2f}%")
    print(f"{'Deep Residual Classifier':<30} | {acc_deep*100:>13.2f}% | {f1_deep*100:>13.2f}%")
    print("=" * 70)
    
    print("\nDetailed Per-Class Classification Report (Deep Residual Classifier):")
    print(classification_report(targets, preds_deep, target_names=CLASSES, digits=4, zero_division=0))
    
    # Save Confusion Matrix
    cm = confusion_matrix(targets, preds_deep)
    plt.figure(figsize=(13, 11))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Quadtree-JEPA Deep Residual Classifier Confusion Matrix", fontsize=13, fontweight='bold')
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
    out_plot = "confusion_matrix_finetuned.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[*] High-resolution confusion matrix saved to: {os.path.abspath(out_plot)}")

if __name__ == "__main__":
    main()
