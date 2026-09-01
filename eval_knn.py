import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from vit_pytorch.vit import ViT
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
K_VALUES = [1, 3, 5, 7, 11, 15]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_knn():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found at '{CHECKPOINT_PATH}'.")
        print("Please wait for 'train_and_evaluate_jepa.py' to finish training first!")
        return

    print("=" * 65)
    print("      k-NN EVALUATION ON FROZEN QUADTREE-JEPA REPRESENTATIONS     ")
    print("=" * 65)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # 1. Initialize and Load Model
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
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Pretrained JEPA backbone successfully loaded.\n")

    # 2. Load Datasets and Extract Frozen Embeddings
    train_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "train"), target_size=TARGET_SIZE)
    test_dataset = LabeledPlantDataset(os.path.join(DATA_DIR, "test"), target_size=TARGET_SIZE)
    class_names = train_dataset.classes

    print("[Step 1/2] Extracting training embeddings (240 images)...")
    train_feats, train_labels = extract_dataset_embeddings(model, train_dataset)
    
    print("[Step 2/2] Extracting test embeddings (60 images)...")
    test_feats, test_labels = extract_dataset_embeddings(model, test_dataset)

    # Convert to NumPy for sklearn k-NN
    X_train = train_feats.numpy()
    y_train = train_labels.numpy()
    X_test = test_feats.numpy()
    y_test = test_labels.numpy()

    # L2 normalize feature vectors for Cosine Metric
    X_train_norm = X_train / (np.linalg.norm(X_train, axis=-1, keepdims=True) + 1e-8)
    X_test_norm = X_test / (np.linalg.norm(X_test, axis=-1, keepdims=True) + 1e-8)

    print("\n" + "=" * 65)
    print("                  k-NN BENCHMARK COMPARISON                      ")
    print("=" * 65)
    print(f"| {'k':<5} | {'Metric':<10} | {'Accuracy':<10} | {'Precision':<11} | {'Recall':<10} | {'F1-Score':<10} |")
    print("-" * 65)

    best_k = 5
    best_f1 = -1
    best_preds = None

    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train_norm, y_train)
        preds = knn.predict(X_test_norm)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_preds = preds

        print(f"| {k:<5} | {'Cosine':<10} | {acc*100:>8.2f}% | {prec*100:>9.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}% |")

    print("-" * 65)
    print(f"\n Optimal k value: k = {best_k} (F1-Score: {best_f1 * 100:.2f}%)")

    # Detailed report for best k
    print("\nDetailed Per-Class Report for Best k-NN (k={}):".format(best_k))
    print(classification_report(y_test, best_preds, target_names=class_names, digits=4, zero_division=0))

    # Confusion Matrix for best k
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Quadtree-JEPA k-NN Confusion Matrix (k={best_k})")
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
    knn_plot_path = "confusion_matrix_knn.png"
    plt.savefig(knn_plot_path, dpi=300)
    plt.close()
    print(f" k-NN Confusion matrix plot saved to: {os.path.abspath(knn_plot_path)}")

if __name__ == "__main__":
    evaluate_knn()
