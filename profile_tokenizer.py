import os
import torch
from torch.utils.data import DataLoader
from dataset import QuadtreeImageDataset
from quadtree_jepa import QuadtreeTokenizer

# Configuration paths
IMAGE_DIR = "./data/training_samples"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"Created empty directory: {IMAGE_DIR}. Drop 5-10 real images (.jpg/.png) inside it before running.")
    exit()

# Instantiate dataset and tokenizer with calibrated thresholds
dataset = QuadtreeImageDataset(image_dir=IMAGE_DIR, target_size=504)
tokenizer = QuadtreeTokenizer(thresholds=[0.28, 0.18, 0.11, 0.0])

print("==================================================")
print("     QUADTREE TOKENIZER SCALE DISTRIBUTION TRACKER")
print("==================================================")

# Loop directly through file paths to extract names while running processing passes
for idx, img_path in enumerate(dataset.image_paths):
    img_name = os.path.basename(img_path)
    
    # Extract item and simulate single-batch dimension loading
    img_tensor = dataset[idx].to(torch.device("cpu"))
    
    with torch.no_grad():
        patches, metadata = tokenizer(img_tensor)
    
    total_tokens = len(metadata)
    scale_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for meta in metadata:
        scale_counts[meta['Z']] += 1
        
    print(f"\nFile: {img_name} [Total Tokens: {total_tokens}]")
    print(f"  └─ Level 0 (64x64) [Context Large]:    {scale_counts[0]} patches ({ (scale_counts[0]/total_tokens)*100:.1f}%)")
    print(f"  └─ Level 1 (32x32) [Context Medium]:   {scale_counts[1]} patches ({ (scale_counts[1]/total_tokens)*100:.1f}%)")
    print(f"  └─ Level 2 (16x16) [Transition]:       {scale_counts[2]} patches ({ (scale_counts[2]/total_tokens)*100:.1f}%)")
    print(f"  └─ Level 3 (8x8)   [Target Micro]:     {scale_counts[3]} patches ({ (scale_counts[3]/total_tokens)*100:.1f}%)")
    
    context_pool = scale_counts[0] + scale_counts[1]
    target_pool = scale_counts[3]
    
    if context_pool == 0:
        print("  ⚠️ WARNING: Context pool is completely empty. Raise thresholds.")
    if target_pool == 0:
        print("  ⚠️ WARNING: Target pool is completely empty. Lower thresholds.")
    if total_tokens > 1200:
        print(f"  ⚠️ MEMORY WARNING: High token allocation ({total_tokens} tokens). Risk of local VRAM spikes.")

print("\nProfiling analysis sequence completed.")