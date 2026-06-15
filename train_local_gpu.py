import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from vit_pytorch.vit import ViT
from quadtree_jepa import QuadtreeJEPA
from dataset import QuadtreeImageDataset
IMAGE_DIR = "./data/training_samples"
dataset = QuadtreeImageDataset(image_dir=IMAGE_DIR, target_size=504)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# Check for CUDA availability (RTX 4060 optimization)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyImageNetMockDataset(Dataset):
    """
    Simulates real image structures instead of complete random noise.
    Uses basic spatial patterns to trigger realistic Quadtree patch splits.
    """
    def __init__(self, num_samples=64, size=504):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create structured gradients to simulate objects/backgrounds
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, self.size), torch.linspace(0, 1, self.size), indexing='ij')
        img = torch.stack([grid_x, grid_y, torch.sin(grid_x * 10)], dim=0)
        # Add slight stochastic variance
        img += torch.randn_like(img) * 0.05
        return torch.clamp(img, 0.0, 1.0)

def print_vram_status():
    """Tracks memory utilization on your RTX 4060."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"  [VRAM] Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB")

# 1. Initialize Bare ViT
base_vit = ViT(
    dim=768,
    depth=6,
    heads=8,
    mlp_dim=1536
).to(device)

# 2. Wrap in QuadtreeJEPA and fine-tune thresholds for realistic image variance
model = QuadtreeJEPA(base_vit=base_vit, embed_dim=768).to(device)
model.tokenizer.thresholds = [0.015, 0.008, 0.003, 0.0]

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
model.train()

# 3. Micro-batching setup for 8GB VRAM hardware limits
dataset = TinyImageNetMockDataset(num_samples=32)
# Batch size of 2 keeps spatial tokens safe from exceeding memory limitations
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Gradient accumulation configurations
accumulation_steps = 8  # Effective Batch Size = 2 * 8 = 16
print(f"Starting hardware configuration validation on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Effective batch size configured to: {dataloader.batch_size * accumulation_steps}")
print_vram_status()

print("\n--- Running GPU Local Profile ---")
optimizer.zero_grad()

for step, images in enumerate(dataloader):
    batch_loss = 0.0
    valid_images_in_step = 0
    
    # Process images individually within the micro-batch to prevent shape conflicts
    for i in range(images.size(0)):
        img = images[i].to(device)
        
        predicted_targets, true_targets = model(img)
        
        if predicted_targets is None or true_targets is None:
            continue
            
        min_len = min(predicted_targets.size(1), true_targets.size(1))
        if min_len == 0:
            continue
            
        # Compute latent reconstruction objective
        loss = F.mse_loss(predicted_targets[:, :min_len, :], true_targets[:, :min_len, :])
        # Scale loss according to accumulation configurations
        loss = loss / accumulation_steps
        loss.backward()
        
        batch_loss += loss.item() * accumulation_steps
        valid_images_in_step += 1
        
    if valid_images_in_step > 0:
        avg_batch_loss = batch_loss / valid_images_in_step
        print(f"Step {step+1}/{len(dataloader)} | Micro-batch Mean Loss: {avg_batch_loss:.6f}")
        
    # Step optimizer after collecting gradients across accumulation sequence
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        model.update_target_encoder(momentum=0.996)
        print(">> Optimizer step executed. Target encoder weights updated via EMA.")
        print_vram_status()

print("\nValidation complete. Your 4060 can cleanly run this training configuration.")