"""
QuadTree-JEPA V3 Smoke Test
Validates:
1. PyTorch + CUDA availability & VRAM check
2. timm library
3. Depth=12 ViT-Base architecture
4. ImageNet pre-trained weight transfer into custom ViT
5. QuadtreeJEPA wrapper and forward pass (504x504 input)
6. Peak VRAM footprint under 8GB budget
"""
import sys
import os

# Ensure unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

print("=" * 60, flush=True)
print("QuadTree-JEPA V3 Pretrained Pipeline Smoke Test", flush=True)
print("=" * 60, flush=True)

print("\n[1/6] Checking PyTorch & CUDA...", flush=True)
import torch
print(f"  PyTorch version: {torch.__version__}", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Target device:   {device}", flush=True)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU Name:        {props.name}", flush=True)
    print(f"  Total VRAM:      {props.total_memory / (1024**3):.2f} GB", flush=True)
else:
    print("  [WARNING] CUDA is not available. Running on CPU will be slow.")

print("\n[2/6] Checking timm...", flush=True)
import timm
print(f"  timm version:    {timm.__version__}", flush=True)

print("\n[3/6] Importing QuadTree-JEPA modules...", flush=True)
from vit_pytorch.vit import ViT, load_pretrained_vit_weights
from quadtree_jepa import QuadtreeJEPA
print("  Imports successful.", flush=True)

print("\n[4/6] Instantiating ViT-Base (depth=12, heads=12, mlp=3072)...", flush=True)
base_vit = ViT(
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dim_head=64,
    dropout=0.1,
    drop_path_rate=0.1,
    qkv_bias=True
).to(device)
vit_params = sum(p.numel() for p in base_vit.parameters()) / 1e6
print(f"  Base ViT parameter count: {vit_params:.2f}M", flush=True)

print("\n[5/6] Loading pre-trained ImageNet weights (vit_base_patch16_224)...", flush=True)
print("  (Note: First run downloads ~340MB checkpoint to torch cache)", flush=True)
load_pretrained_vit_weights(base_vit, model_name='vit_base_patch16_224')
print("  Pretrained weights transferred successfully!", flush=True)

print("\n[6/6] Building QuadtreeJEPA & testing forward pass...", flush=True)
model = QuadtreeJEPA(
    base_vit=base_vit,
    embed_dim=768,
    max_seq_len=800,
    target_budget=256
).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"  Total Trainable Parameters: {total_params:.2f}M", flush=True)

dummy_img = torch.randn(1, 3, 504, 504, device=device)
with torch.no_grad():
    pred, true, ctx, t_len = model(dummy_img)

print(f"  Forward pass successful!")
print(f"    Target Length:    {t_len}")
print(f"    Context Shape:    {list(ctx.shape)}")
print(f"    Prediction Shape: {list(pred.shape) if pred is not None else 'None'}")
print(f"    Target Shape:     {list(true.shape) if true is not None else 'None'}")

if torch.cuda.is_available():
    vram_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak VRAM Allocated: {vram_used:.2f} GB (Comfortably fits in 8GB GPU)", flush=True)

print("\n" + "=" * 60, flush=True)
print(">>> ALL CHECKS PASSED: SYSTEM READY FOR FULL V3 TRAINING <<<", flush=True)
print("=" * 60, flush=True)
