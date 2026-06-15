import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import QuadtreeImageDataset
from quadtree_jepa import QuadtreeJEPA
from vit_pytorch.vit import ViT

# Target your local RTX 4060
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==================================================")
# Scale-aware pre-training loop activation sequence
print("        INITIALIZING QUADTREE-JEPA ENGINE         ")
print("==================================================")

# 1. Instantiate the stripped transformer core
base_vit = ViT(
    dim=768,
    depth=6,
    heads=8,
    mlp_dim=1536
).to(device)

# 2. Wrap into our custom asymmetric masking objective
model = QuadtreeJEPA(base_vit=base_vit, embed_dim=768).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# 3. Pull raw samples from your local directory
IMAGE_DIR = "./data/training_samples"
dataset = QuadtreeImageDataset(image_dir=IMAGE_DIR, target_size=504)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Gradient Accumulation simulates a larger batch size on low VRAM
accumulation_steps = 16 
epochs = 10

print(f"\n[Hardware Status] Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"[Dataset Status] Found {len(dataset)} valid training images.")
print(f"[Hyperparameters] Target Sequence Cap: 800 | Effective Batch Size: {accumulation_steps}\n")

model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    running_loss = 0.0
    processed_images = 0
    
    print(f"--- Epoch {epoch+1}/{epochs} ---")
    
    for step, img_tensor in enumerate(dataloader):
        # Remove batch dimension for tokenizer loop processing
        img_single = img_tensor.squeeze(0).to(device)
        
        # Pass image through dynamic quadtree splits and target projections
        predicted_targets, true_targets = model(img_single)
        
        if predicted_targets is None or true_targets is None:
            continue
            
        # Match dimensions across sequence sequences
        min_len = min(predicted_targets.size(1), true_targets.size(1))
        if min_len == 0:
            continue
            
        # Latent space Mean Squared Error minimization objective
        loss = F.mse_loss(predicted_targets[:, :min_len, :], true_targets[:, :min_len, :])
        
        # Scale loss to adjust for gradient accumulation pacing
        loss = loss / accumulation_steps
        loss.backward()
        
        running_loss += loss.item() * accumulation_steps
        processed_images += 1
        
        # Trigger gradient step when accumulation criteria is satisfied
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
            # Slowly inject context parameters into target network
            model.update_target_encoder(momentum=0.996)
            
            current_avg_loss = running_loss / max(processed_images, 1)
            print(f"  Step [{step+1}/{len(dataloader)}] | Accumulated Latent MSE Loss: {current_avg_loss:.6f}")

print("\nPre-training phase complete. Weights are successfully updating.")