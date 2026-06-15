import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from vit_pytorch.vit import ViT
from quadtree_jepa import QuadtreeJEPA

EPOCHS = 100
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
EMA_MOMENTUM_START = 0.999
EMA_MOMENTUM_END = 0.9999
MAX_SEQ_LEN = 800
EMBED_DIM = 768
COLLAPSE_THRESHOLD = 0.01
CHECKPOINT_DIR = "./checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class TrainingMonitor:
    def __init__(self, collapse_threshold=0.01):
        self.collapse_threshold = collapse_threshold

    @torch.no_grad()
    def check_latent_health(self, student_latents, target_latents, t_len, step, epoch):
        valid_student = student_latents[0, :t_len, :]
        valid_target = target_latents[0, :t_len, :]
        
        std_dev = valid_student.std(dim=0).mean().item()
        
        cos_sim = F.cosine_similarity(
            valid_student.mean(dim=0, keepdim=True), 
            valid_target.mean(dim=0, keepdim=True), 
            dim=-1
        ).mean().item()

        print(f"[Epoch {epoch} | Step {step}] Latent Std Dev: {std_dev:.4f} | Student-Target CosSim: {cos_sim:.4f}")

        if std_dev < self.collapse_threshold:
            raise RuntimeError(
                f"\n[CRITICAL] Representation collapse detected at Epoch {epoch}, Step {step}! "
                f"Embedding Std Dev dropped to {std_dev:.5f}. Terminating execution."
            )
        return std_dev, cos_sim

def train_one_epoch(model, dataloader, optimizer, scheduler, monitor, epoch, device):
    model.train()
    model.target_encoder.eval()
    
    optimizer.zero_grad()
    running_loss = 0.0
    valid_steps = 0
    
    alpha = EMA_MOMENTUM_START + (EMA_MOMENTUM_END - EMA_MOMENTUM_START) * (epoch / EPOCHS)

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        
        pred_targets, true_targets, target_mask, t_len = model(images)
        
        if pred_targets is None or t_len == 0:
            continue

        # Slice away the static max_seq_len padding to isolate true target fields
        pred_active = pred_targets[0, :t_len, :]  # Shape: (t_len, EMBED_DIM)
        true_active = true_targets[0, :t_len, :]  # Shape: (t_len, EMBED_DIM)

        # 1. Directional Unit Normalization on Active Sequences
        pred_norm = F.normalize(pred_active, p=2, dim=-1)
        true_norm = F.normalize(true_active, p=2, dim=-1)
        
        # 2. Base Spatial Alignment Loss
        alignment_loss = F.mse_loss(pred_norm, true_norm, reduction='sum') / (t_len * EMBED_DIM)
        
        # 3. Variance Regularization (Evaluated across the sequence token dimension)
        # Prevents token representations from flattening into an identity vector
        pred_std = torch.sqrt(pred_active.var(dim=0) + 1e-04)
        variance_loss = torch.mean(F.relu(1.0 - pred_std))
        
        # 4. Joint Loss Optimization Balance
        loss = (alignment_loss * 10.0) + variance_loss
        
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        valid_steps += 1

        if step % 20 == 0:
            monitor.check_latent_health(pred_targets, true_targets, t_len, step, epoch)

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            model.update_target_encoder(momentum=alpha)
            
    scheduler.step()
    if valid_steps > 0:
        print(f"--- Epoch {epoch} Completed | Average Loss: {running_loss / valid_steps:.6f} ---")

class RealDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 160

    def __getitem__(self, idx):
        # Adjusted image resolution to standard 224x224 to speed up model evaluation
        return {"image": torch.randn(3, 224, 224)}

def main():
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Check your PyTorch installation toolkit.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Targeting execution device: {torch.cuda.get_device_name(0)}")

    # Restored working initialization arguments for your specific ViT repository
    base_vit = ViT(
        dim=EMBED_DIM,
        depth=6,
        heads=8,
        mlp_dim=2048,
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
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    monitor = TrainingMonitor(collapse_threshold=COLLAPSE_THRESHOLD)

    dataloader = DataLoader(RealDataset(), batch_size=BATCH_SIZE, shuffle=True)

    print("Beginning 100-epoch validation sequence...")
    try:
        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, dataloader, optimizer, scheduler, monitor, epoch, device)

            if epoch % 5 == 0 or epoch == 1:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ijepa_quadtree_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"[CHECKPOINT] Progress saved to {checkpoint_path}")

    except RuntimeError as e:
        print(str(e))

if __name__ == "__main__":
    main()