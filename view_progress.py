import os
import glob
import torch

CHECKPOINT_DIR = "./checkpoints"

def check_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_checkpoint_info():
    checkpoint_files = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "ijepa_quadtree_epoch_*.pt")),
        key=lambda x: [int(c) if c.isdigit() else c for c in os.path.basename(x).split('_')]
    )
    
    if not checkpoint_files:
        print(f"No checkpoint files found in directory: '{CHECKPOINT_DIR}'")
        return

    print(f"{'-'*85}")
    print(f"| {'File Name':<30} | {'Epoch':<7} | {'File Size':<12} | {'Keys Saved in State Dict':<22} |")
    print(f"{'-'*85}")

    for file_path in checkpoint_files:
        file_name = os.path.basename(file_path)
        file_size = check_file_size(file_path)
        
        try:
            # map_location='cpu' ensures it loads safely even if CUDA is busy or inactive
            checkpoint = torch.load(file_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'N/A')
            
            # Count top-level tracking tracking keys inside the checkpoint dictionary
            saved_keys = ", ".join(list(checkpoint.keys()))
            if len(saved_keys) > 22:
                saved_keys = saved_keys[:19] + "..."
                
            print(f"| {file_name:<30} | {epoch:<7} | {file_size:<12} | {saved_keys:<22} |")
        except Exception as e:
            print(f"| {file_name:<30} | {'Corrupt':<7} | {file_size:<12} | {'Error loading file':<22} |")
            
    print(f"{'-'*85}")

if __name__ == "__main__":
    get_checkpoint_info()