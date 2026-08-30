import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from vit_pytorch.vit import ViT
from vit_pytorch.simple_vit import SimpleViT
from quadtree_jepa import QuadtreeJEPA
from train_and_evaluate_jepa import LabeledPlantDataset, extract_dataset_embeddings

# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "./checkpoints/jepa_plant_75epochs.pt"
DATA_DIR = "./data/plant_dataset"
TARGET_SIZE = 504
EMBED_DIM = 768
MAX_SEQ_LEN = 800

# Fractions / sample counts to test per class
SAMPLES_PER_CLASS_LIST = [5, 10, 20, 40, 80]  # 5 (~6%), 10 (~12.5%), 20 (~25%), 40 (~50%), 80 (100%)
SUPERVISED_EPOCHS = 35
PROBE_EPOCHS = 40
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# SUPERVISED VIT MODEL DEFINITION
# ==========================================
class SupervisedViTClassifier(nn.Module):
    """
    Standard Supervised Vision Transformer baseline with patch embedding
    and identical transformer backbone capacity (dim=768, depth=6, heads=8).
    """
    def __init__(self, num_classes=3, image_size=504, patch_size=28, embed_dim=768, depth=6, heads=8, mlp_dim=1536):
        super().__init__()
        self.vit = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, x):
        return self.vit(x)

# ==========================================
# HELPER: STRATIFIED SUBSET SAMPLER
# ==========================================
def get_stratified_subset_indices(dataset, samples_per_class):
    """
    Selects exactly `samples_per_class` indices for each class in the dataset.
    """
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in sorted(class_to_indices.items()):
        shuffled = indices.copy()
        random.shuffle(shuffled)
        k = min(samples_per_class, len(shuffled))
        selected_indices.extend(shuffled[:k])

    random.shuffle(selected_indices)
    return selected_indices

# ==========================================
# TRAIN SUPERVISED VIT FROM SCRATCH
# ==========================================
def train_and_eval_supervised_vit(train_dataset, subset_indices, test_loader, num_classes, epochs=35):
    subset = Subset(train_dataset, subset_indices)
    loader = DataLoader(subset, batch_size=8, shuffle=True)

    model = SupervisedViTClassifier(
        num_classes=num_classes,
        image_size=TARGET_SIZE,
        patch_size=28,
        embed_dim=EMBED_DIM,
        depth=6,
        heads=8,
        mlp_dim=1536
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluate on full test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)[2]
    return acc, f1

# ==========================================
# TRAIN JEPA LINEAR PROBE ON FROZEN FEATURES
# ==========================================
def train_and_eval_jepa_probe(all_train_feats, all_train_labels, subset_indices, test_feats, test_labels, num_classes, epochs=40):
    sub_feats = all_train_feats[subset_indices]
    sub_labels = all_train_labels[subset_indices]

    linear_head = nn.Linear(EMBED_DIM, num_classes).to(device)
    optimizer = torch.optim.Adam(linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(sub_feats, sub_labels), batch_size=min(16, len(sub_feats)), shuffle=True)

    linear_head.train()
    for epoch in range(1, epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = linear_head(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluate on full test set
    linear_head.eval()
    with torch.no_grad():
        logits = linear_head(test_feats.to(device))
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        targets = test_labels.numpy()

    acc = accuracy_score(targets, preds)
    f1 = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)[2]
    return acc, f1

# ==========================================
# MAIN BENCHMARK RUNNER
# ==========================================
def main():
    set_seed(SEED)
    print("=" * 70)
    print("     LABEL-EFFICIENCY BENCHMARK: SUPERVISED ViT vs QUADTREE-JEPA    ")
    print("=" * 70)

    # 1. Load Datasets
    train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE)
    test_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "test"), target_size=TARGET_SIZE)
    num_classes = len(train_dataset.classes)
    total_train = len(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Dataset: Plant Disease ({num_classes} classes: {train_dataset.classes})")
    print(f"Total Training Images: {total_train} | Total Test Images: {len(test_dataset)}\n")

    # 2. Load Pretrained Quadtree-JEPA and Extract Frozen Features
    print(f"[Step 1/3] Loading pretrained Quadtree-JEPA from: {CHECKPOINT_PATH}...")
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
    jepa_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    jepa_model.eval()

    print("[Step 2/3] Extracting frozen latent embeddings for all images...")
    all_train_feats, all_train_labels = extract_dataset_embeddings(jepa_model, train_dataset)
    test_feats, test_labels = extract_dataset_embeddings(jepa_model, test_dataset)
    print(" Feature extraction complete.\n")

    # 3. Benchmark Across Label Fractions
    print("[Step 3/3] Running benchmark across label budget levels...")
    results = []

    for spc in SAMPLES_PER_CLASS_LIST:
        total_samples = min(spc * num_classes, total_train)
        pct = (total_samples / total_train) * 100.0
        print(f"\n--- Testing with {spc} samples/class ({total_samples} total images = {pct:.1f}% labeled data) ---")

        subset_idx = get_stratified_subset_indices(train_dataset, spc)

        # Train & evaluate JEPA linear probe
        jepa_acc, jepa_f1 = train_and_eval_jepa_probe(
            all_train_feats, all_train_labels, subset_idx, test_feats, test_labels, num_classes, epochs=PROBE_EPOCHS
        )

        # Train & evaluate Supervised ViT
        sup_acc, sup_f1 = train_and_eval_supervised_vit(
            train_dataset, subset_idx, test_loader, num_classes, epochs=SUPERVISED_EPOCHS
        )

        print(f"  [Result] Supervised ViT Accuracy: {sup_acc * 100:.2f}% | F1: {sup_f1 * 100:.2f}%")
        print(f"  [Result] Quadtree-JEPA Accuracy:   {jepa_acc * 100:.2f}% | F1: {jepa_f1 * 100:.2f}%")

        results.append({
            'spc': spc,
            'total_samples': total_samples,
            'pct': pct,
            'sup_acc': sup_acc * 100,
            'sup_f1': sup_f1 * 100,
            'jepa_acc': jepa_acc * 100,
            'jepa_f1': jepa_f1 * 100,
        })

    # 4. Print Comparison Table
    print("\n" + "=" * 78)
    print(f"| {'Labels/Class':<13} | {'Total Labels':<12} | {'% Data':<8} | {'Supervised ViT Acc':<19} | {'Quadtree-JEPA Acc':<18} |")
    print("-" * 78)
    for r in results:
        print(f"| {r['spc']:<13} | {r['total_samples']:<12} | {r['pct']:>5.1f}%  | {r['sup_acc']:>16.2f}%   | {r['jepa_acc']:>15.2f}%   |")
    print("=" * 78)

    # 5. Plot Publication-Quality Graph
    pcts = [r['pct'] for r in results]
    sup_accs = [r['sup_acc'] for r in results]
    jepa_accs = [r['jepa_acc'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(pcts, jepa_accs, marker='o', linewidth=2.5, markersize=8, color='#2563EB', label='Quadtree-JEPA (Frozen SSL Backbone + Probe)')
    plt.plot(pcts, sup_accs, marker='s', linewidth=2.5, markersize=8, color='#DC2626', linestyle='--', label='Supervised ViT (Trained from Scratch)')

    for i, r in enumerate(results):
        plt.annotate(f"{r['jepa_acc']:.1f}%", (pcts[i], jepa_accs[i] + 1.5), fontsize=9, fontweight='bold', color='#1E40AF', ha='center')
        plt.annotate(f"{r['sup_acc']:.1f}%", (pcts[i], sup_accs[i] - 3.0), fontsize=9, fontweight='bold', color='#991B1B', ha='center')

    plt.title("Label-Efficiency Benchmark: Quadtree-JEPA vs. Supervised ViT", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Percentage of Labeled Training Data (%)", fontsize=12, labelpad=10)
    plt.ylabel("Test Set Classification Accuracy (%)", fontsize=12, labelpad=10)
    plt.ylim(20, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.95)
    plt.tight_layout()

    plot_path = "label_efficiency_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n Benchmark curve saved successfully to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    main()
