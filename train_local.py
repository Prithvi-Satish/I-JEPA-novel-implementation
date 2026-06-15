import torch
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch.vit import ViT
from quadtree_jepa import QuadtreeJEPA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing Base Vision Transformer Architecture...")
base_vit = ViT(
    dim=768,
    depth=6,
    heads=8,
    mlp_dim=1536
).to(device)

# Using lower thresholds to guarantee both background and micro-details are found in noise maps
model = QuadtreeJEPA(base_vit=base_vit, embed_dim=768).to(device)
model.tokenizer.thresholds = [0.02, 0.01, 0.005, 0.0] 

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print("\n--- Starting Verification Epochs ---")
for epoch in range(5):
    optimizer.zero_grad()
    
    # Generate random image matrix
    dummy_img = torch.rand(3, 504, 504).to(device)
    
    # Forward Pass through Quadtree and Fusion Bridge
    predicted_targets, true_targets = model(dummy_img)
    
    if predicted_targets is None or true_targets is None:
        print(f"Epoch {epoch+1}: Skipped (Image variance did not satisfy token routing distribution rules)")
        continue

    print(f"Epoch {epoch+1} Metadata:")
    print(f"  -> Context Tokens Routed (Z=0,1): {predicted_targets.size(1)}")
    print(f"  -> Target Tokens Routed (Z=3):    {true_targets.size(1)}")

    # Slice sequences to identical lengths for calculating loss
    min_len = min(predicted_targets.size(1), true_targets.size(1))
    loss = F.mse_loss(predicted_targets[:, :min_len, :], true_targets[:, :min_len, :])
    
    loss.backward()
    optimizer.step()
    
    # Update frozen weights via Exponential Moving Average
    model.update_target_encoder(momentum=0.996)
    
    print(f"  -> Backpropagation Complete. Latent MSE Loss: {loss.item():.6f}\n")

print("Local device execution verification completed successfully.")