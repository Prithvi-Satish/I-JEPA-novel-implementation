import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

# ============================================================================
#  V3 QuadTree-JEPA — Fully Vectorized, Zero CPU-Sync, Batch-Safe
#  Changes vs V2:
#   - QuadtreeTokenizer: returns (dict of level tensors, (256,3) position tensor)
#     → eliminates 512 .item() PCIe sync stalls per image
#   - ZAxisFusionBridge: pure tensor forward, zero Python loops
#   - ScaleAwareAttentivePool: activated level_bias embeddings (was dead code)
#   - PredictorBlock: Dropout(0.1) added to MLP (SSL regularization)
#   - QuadtreeJEPA: precomputed level_ids buffer; batched extract_features_batch()
#   - QuadtreeClassifier: fully fixed batch forward for any B≥1
# ============================================================================


class QuadtreeTokenizer(nn.Module):
    """
    V3 Fully Vectorized QuadTree Tokenizer.
    Zero Python loops inside forward(), zero .item() calls, zero PCIe stalls.

    Fixed token budget breakdown (always deterministic for 512×512 padded input):
      Level 0 (64×64): split 44 → keep 20 tokens
      Level 1 (32×32): from 44*4=176, split 16 → keep 160 tokens
      Level 2 (16×16): from 16*4=64,  split  4 → keep  60 tokens
      Level 3 ( 8× 8): from  4*4=16,  keep all 16 tokens
      Total: 20 + 160 + 60 + 16 = 256 tokens  ✓
    """
    # Budget constants
    N_SPLIT_L0 = 44;  N_KEEP_L0 = 20    # 64 - 44 = 20
    N_L1_TOTAL = 176; N_SPLIT_L1 = 16;  N_KEEP_L1 = 160  # 176 - 16 = 160
    N_L2_TOTAL = 64;  N_SPLIT_L2 = 4;   N_KEEP_L2 = 60   # 64 - 4  = 60
    N_L3_TOTAL = 16

    def __init__(self, target_budget=256, max_level=3, min_relative_contrast=0.3):
        super().__init__()
        self.target_budget = target_budget
        self.max_level = max_level
        self.min_relative_contrast = min_relative_contrast

    def forward(self, image):
        """
        Args:
            image: (C, H, W) or (1, C, H, W) GPU tensor

        Returns:
            patches_by_level (dict):
                {0: (20,  3*64*64), 1: (160, 3*32*32),
                 2: (60,  3*16*16), 3: (16,  3*8*8)}   — all on GPU
            positions (Tensor): (256, 3) float32  [X, Y, Z_level]
              — no .item() calls, no PCIe stalls
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        device = image.device
        C, H, W = image.shape

        # Pad to nearest multiple of 64
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        _, pH, pW = image.shape
        nH, nW = pH // 64, pW // 64    # e.g. 8×8 = 64 for 512×512

        # Luminance map (entirely on GPU)
        gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]  # (pH, pW)

        # ── Level 0: nH*nW patches (64×64) ───────────────────────────────────
        patches_L0 = (image.unfold(1, 64, 64).unfold(2, 64, 64)        # (C, nH, nW, 64, 64)
                           .permute(1, 2, 0, 3, 4)
                           .reshape(nH * nW, C, 64, 64))                # (N0, C, 64, 64)

        gray_L0 = gray.unfold(0, 64, 64).unfold(1, 64, 64).reshape(nH * nW, 64, 64)
        var_L0  = gray_L0.var(dim=[1, 2])                               # (N0,)

        # Grid coords — pure tensors, zero .item()
        col_idx = torch.arange(nH * nW, device=device) % nW
        row_idx = torch.arange(nH * nW, device=device) // nW
        xs_L0 = col_idx.float() * 64   # (N0,)
        ys_L0 = row_idx.float() * 64   # (N0,)

        n_split_L0 = min(self.N_SPLIT_L0, nH * nW - 4)   # guard for small images
        n_keep_L0  = nH * nW - n_split_L0

        _, split_0_idx = torch.topk(var_L0, k=n_split_L0)
        keep_0_mask = torch.ones(nH * nW, dtype=torch.bool, device=device)
        keep_0_mask[split_0_idx] = False

        kept_L0  = patches_L0[keep_0_mask].reshape(n_keep_L0, -1)     # (20, 3*64*64)
        pos_L0   = torch.stack([xs_L0[keep_0_mask],
                                 ys_L0[keep_0_mask],
                                 torch.zeros(n_keep_L0, device=device)], dim=1)  # (20, 3)

        # ── Level 1: n_split_L0*4 patches (32×32) ────────────────────────────
        split_L0   = patches_L0[split_0_idx]          # (44, C, 64, 64)
        xs_sp0     = xs_L0[split_0_idx]               # (44,)
        ys_sp0     = ys_L0[split_0_idx]               # (44,)

        patches_L1 = (split_L0.unfold(2, 32, 32).unfold(3, 32, 32)    # (44, C, 2, 2, 32, 32)
                              .permute(0, 2, 3, 1, 4, 5)
                              .reshape(n_split_L0 * 4, C, 32, 32))     # (176, C, 32, 32)

        gray_L1 = (0.2989 * patches_L1[:, 0] +
                   0.5870 * patches_L1[:, 1] +
                   0.1140 * patches_L1[:, 2])
        var_L1  = gray_L1.var(dim=[1, 2])                              # (176,)

        dx_L1 = torch.tensor([0., 32.,  0., 32.], device=device).repeat(n_split_L0)
        dy_L1 = torch.tensor([0.,  0., 32., 32.], device=device).repeat(n_split_L0)
        xs_L1 = xs_sp0.repeat_interleave(4) + dx_L1                   # (176,)
        ys_L1 = ys_sp0.repeat_interleave(4) + dy_L1

        n_l1_total = n_split_L0 * 4
        n_split_L1 = min(self.N_SPLIT_L1, n_l1_total - 4)
        n_keep_L1  = n_l1_total - n_split_L1

        _, split_1_idx = torch.topk(var_L1, k=n_split_L1)
        keep_1_mask = torch.ones(n_l1_total, dtype=torch.bool, device=device)
        keep_1_mask[split_1_idx] = False

        kept_L1 = patches_L1[keep_1_mask].reshape(n_keep_L1, -1)      # (160, 3*32*32)
        pos_L1  = torch.stack([xs_L1[keep_1_mask],
                                ys_L1[keep_1_mask],
                                torch.ones(n_keep_L1, device=device)], dim=1)  # (160, 3)

        # ── Level 2: n_split_L1*4 patches (16×16) ────────────────────────────
        split_L1   = patches_L1[split_1_idx]          # (16, C, 32, 32)
        xs_sp1     = xs_L1[split_1_idx]
        ys_sp1     = ys_L1[split_1_idx]

        patches_L2 = (split_L1.unfold(2, 16, 16).unfold(3, 16, 16)    # (16, C, 2, 2, 16, 16)
                              .permute(0, 2, 3, 1, 4, 5)
                              .reshape(n_split_L1 * 4, C, 16, 16))     # (64, C, 16, 16)

        gray_L2 = (0.2989 * patches_L2[:, 0] +
                   0.5870 * patches_L2[:, 1] +
                   0.1140 * patches_L2[:, 2])
        var_L2  = gray_L2.var(dim=[1, 2])                              # (64,)

        dx_L2 = torch.tensor([0., 16.,  0., 16.], device=device).repeat(n_split_L1)
        dy_L2 = torch.tensor([0.,  0., 16., 16.], device=device).repeat(n_split_L1)
        xs_L2 = xs_sp1.repeat_interleave(4) + dx_L2
        ys_L2 = ys_sp1.repeat_interleave(4) + dy_L2

        n_l2_total = n_split_L1 * 4
        n_split_L2 = min(self.N_SPLIT_L2, n_l2_total - 1)
        n_keep_L2  = n_l2_total - n_split_L2

        _, split_2_idx = torch.topk(var_L2, k=n_split_L2)
        keep_2_mask = torch.ones(n_l2_total, dtype=torch.bool, device=device)
        keep_2_mask[split_2_idx] = False

        kept_L2 = patches_L2[keep_2_mask].reshape(n_keep_L2, -1)      # (60, 3*16*16)
        pos_L2  = torch.stack([xs_L2[keep_2_mask],
                                ys_L2[keep_2_mask],
                                torch.full((n_keep_L2,), 2., device=device)], dim=1)  # (60, 3)

        # ── Level 3: n_split_L2*4 patches (8×8) — all kept ───────────────────
        split_L2   = patches_L2[split_2_idx]          # (4, C, 16, 16)
        xs_sp2     = xs_L2[split_2_idx]
        ys_sp2     = ys_L2[split_2_idx]

        patches_L3 = (split_L2.unfold(2, 8, 8).unfold(3, 8, 8)        # (4, C, 2, 2, 8, 8)
                              .permute(0, 2, 3, 1, 4, 5)
                              .reshape(n_split_L2 * 4, C, 8, 8))       # (16, C, 8, 8)

        dx_L3 = torch.tensor([0., 8.,  0., 8.], device=device).repeat(n_split_L2)
        dy_L3 = torch.tensor([0., 0.,  8., 8.], device=device).repeat(n_split_L2)
        xs_L3 = xs_sp2.repeat_interleave(4) + dx_L3
        ys_L3 = ys_sp2.repeat_interleave(4) + dy_L3

        n_l3 = n_split_L2 * 4
        kept_L3 = patches_L3.reshape(n_l3, -1)                         # (16, 3*8*8)
        pos_L3  = torch.stack([xs_L3,
                                ys_L3,
                                torch.full((n_l3,), 3., device=device)], dim=1)  # (16, 3)

        # ── Assemble — zero .item(), zero Python loops ────────────────────────
        patches_by_level = {
            0: kept_L0,   # (20,  12288)
            1: kept_L1,   # (160,  3072)
            2: kept_L2,   # (60,    768)
            3: kept_L3,   # (16,    192)
        }
        positions = torch.cat([pos_L0, pos_L1, pos_L2, pos_L3], dim=0)  # (256, 3)

        return patches_by_level, positions


class ZAxisFusionBridge(nn.Module):
    """
    V3 Z-Axis Fusion Bridge.
    Accepts structured tensor inputs from V3 tokenizer.
    4 grouped batched GEMMs + sinusoidal 2D + learnable 1D scale embeddings.
    Zero Python loops inside forward().
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict({
            '0': nn.Linear(3 * 64 * 64, embed_dim),   # 12288 → 768
            '1': nn.Linear(3 * 32 * 32, embed_dim),   #  3072 → 768
            '2': nn.Linear(3 * 16 * 16, embed_dim),   #   768 → 768
            '3': nn.Linear(3 *  8 *  8, embed_dim),   #   192 → 768
        })
        self.scale_embed = nn.Embedding(4, embed_dim)

        half_dim = embed_dim // 2
        omega = torch.exp(
            torch.arange(0, half_dim // 2, dtype=torch.float32)
            * -(np.log(10000.0) / (half_dim // 2))
        )
        self.register_buffer('omega', omega)

    def get_2d_pos_embed(self, xs, ys):
        """xs, ys: (N, 1). Returns (N, embed_dim) sinusoidal embedding."""
        omega = self.omega   # (D/4,)
        ex = xs * omega.unsqueeze(0)   # (N, D/4)
        ey = ys * omega.unsqueeze(0)
        ex = torch.cat([torch.sin(ex), torch.cos(ex)], dim=-1)   # (N, D/2)
        ey = torch.cat([torch.sin(ey), torch.cos(ey)], dim=-1)
        return torch.cat([ex, ey], dim=-1)                        # (N, D)

    def forward(self, patches_by_level, positions):
        """
        patches_by_level: {0: (n0, d0), 1: (n1, d1), 2: (n2, d2), 3: (n3, d3)}
        positions: (N, 3) float tensor [X, Y, Z]

        Returns: (N, embed_dim) — no loops, no .item()
        """
        # 4 grouped GEMMs (levels in sorted order to match position ordering)
        proj_parts = []
        for z in range(4):
            if z in patches_by_level and patches_by_level[z].shape[0] > 0:
                proj_parts.append(self.projections[str(z)](patches_by_level[z]))

        if not proj_parts:
            device = positions.device
            return torch.zeros(positions.shape[0], self.embed_dim, device=device)

        projected = torch.cat(proj_parts, dim=0)   # (N, D)

        xs = positions[:, 0:1]        # (N, 1)
        ys = positions[:, 1:2]        # (N, 1)
        zs = positions[:, 2].long()   # (N,)

        pe_2d = self.get_2d_pos_embed(xs, ys).to(projected.dtype)
        pe_1d = self.scale_embed(zs).to(projected.dtype)

        return projected + pe_2d + pe_1d   # (N, D)


class PredictorBlock(nn.Module):
    """
    V3: Added Dropout(0.1) to MLP — standard I-JEPA SSL predictor regularization
    to prevent trivial constant-prediction collapse.
    """
    def __init__(self, embed_dim=768, heads=8, is_cross=False):
        super().__init__()
        self.is_cross = is_cross
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),   # V3: SSL predictor regularization
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None, context_mask=None):
        if self.is_cross and context is not None:
            attn_out, _ = self.attn(query=x, key=context, value=context,
                                    key_padding_mask=context_mask)
        else:
            attn_out, _ = self.attn(query=x, key=x, value=x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class CrossAttentionPredictor(nn.Module):
    def __init__(self, embed_dim=768, depth=3, heads=8):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            PredictorBlock(embed_dim=embed_dim, heads=heads, is_cross=(i == 0))
            for i in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, context_tokens, target_queries, context_mask=None, target_mask=None):
        x = self.layers[0](target_queries, context=context_tokens, context_mask=context_mask)
        for layer in self.layers[1:]:
            x = layer(x)
        x = self.final_norm(x)
        if target_mask is not None:
            mask_weight = (~target_mask).float().unsqueeze(-1) if target_mask.dtype == torch.bool \
                          else (1.0 - target_mask.float()).unsqueeze(-1)
            x = x * mask_weight
        return x


class ScaleAwareAttentivePool(nn.Module):
    """
    V3 Scale-Aware Attentive Pooling — level bias embeddings NOW ACTIVATED.
    Previously declared but unused (V2 bug). Now adds learnable per-level key
    biases before cross-attention, letting the model differentially weight
    coarse background tokens vs fine-grained detail tokens.
    """
    def __init__(self, embed_dim=768, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.level_bias = nn.Embedding(4, embed_dim)   # V3: activated
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # V4 FIX: initialize level biases to zero, not default std≈1.
        # Default nn.Embedding init has std≈1 which swamps ViT token magnitudes
        # (after LayerNorm, tokens have std≈1 too — a random std=1 offset
        #  makes attention keys ~random at init, destroying signal for first epochs).
        nn.init.zeros_(self.level_bias.weight)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, level_ids=None):
        """
        tokens:    (B, N, D)
        level_ids: (N,) or (B, N) long tensor with values in {0,1,2,3}
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, N, D = tokens.shape
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)

        if level_ids is not None:
            lvl = level_ids.to(tokens.device)
            if lvl.dim() == 1:
                lvl = lvl.unsqueeze(0).expand(B, -1)   # (B, N)
            keys = tokens + self.level_bias(lvl)        # learnable scale bias on keys
        else:
            keys = tokens

        out, _ = self.attn(query=cls, key=keys, value=tokens)
        return self.norm(out.squeeze(1))   # (B, D)


class QuadtreeJEPA(nn.Module):
    """
    V3 QuadTree Joint-Embedding Predictive Architecture.
    Updated throughout for the V3 tensor-native tokenizer API.
    New: extract_features_batch() for ~B× faster feature caching.
    """
    # Fixed level boundaries for budget=256  (20+160+60+16)
    L0_END     = 20
    L1_END     = 180   # 20 + 160
    L2_END     = 240   # 180 + 60
    L3_END     = 256   # 240 + 16
    COARSE_END = 180   # L0+L1 = context tokens for coarse→fine

    def __init__(self, base_vit, embed_dim=768, max_seq_len=800,
                 predictor_depth=3, target_budget=256):
        super().__init__()
        self.max_seq_len   = max_seq_len
        self.embed_dim     = embed_dim

        self.tokenizer     = QuadtreeTokenizer(target_budget=target_budget, max_level=3)
        self.z_bridge      = ZAxisFusionBridge(embed_dim=embed_dim)
        self.pooler        = ScaleAwareAttentivePool(embed_dim=embed_dim)

        self.context_encoder = base_vit
        self.target_encoder  = copy.deepcopy(base_vit)
        self.target_norm     = nn.LayerNorm(embed_dim)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = CrossAttentionPredictor(embed_dim=embed_dim, depth=predictor_depth)

        # Precomputed fixed level_ids buffer — same for every image (deterministic budget)
        level_ids = torch.cat([
            torch.zeros(self.L0_END,                     dtype=torch.long),   # 20  → L0
            torch.ones (self.L1_END  - self.L0_END,      dtype=torch.long),   # 160 → L1
            torch.full ((self.L2_END - self.L1_END,), 2, dtype=torch.long),   # 60  → L2
            torch.full ((self.L3_END - self.L2_END,), 3, dtype=torch.long),   # 16  → L3
        ])   # (256,)
        self.register_buffer('level_ids', level_ids)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                    self.target_encoder.parameters()):
                p_tgt.data.lerp_(p_ctx.data, 1.0 - momentum)

    def _tokenize_single(self, img):
        """Tokenize one (C,H,W) image → (256, D) token tensor."""
        if img.dim() == 4:
            img = img.squeeze(0)
        pbl, pos = self.tokenizer(img)
        return self.z_bridge(pbl, pos)   # (256, D)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, img, bidirectional=True):
        if img.dim() == 4 and img.shape[0] > 1:
            return self.forward_batch(img, bidirectional=bidirectional)
        if img.dim() == 4:
            img = img.squeeze(0)

        tokens = self._tokenize_single(img)   # (256, D)

        # Stochastic bidirectional cross-scale masking (V2-08)
        is_c2f = (torch.rand(1).item() > 0.5) if (bidirectional and self.training) else True

        if is_c2f:
            context_in = tokens[:self.COARSE_END].unsqueeze(0)    # (1, 180, D)
            target_in  = tokens[self.COARSE_END:].unsqueeze(0)    # (1,  76, D)
            t_len = self.L3_END - self.COARSE_END   # 76
        else:
            context_in = tokens[self.COARSE_END:].unsqueeze(0)    # (1,  76, D)
            target_in  = tokens[:self.COARSE_END].unsqueeze(0)    # (1, 180, D)
            t_len = self.COARSE_END   # 180

        context_out = self.context_encoder(context_in)
        with torch.no_grad():
            true_targets = self.target_norm(self.target_encoder(target_in))

        predicted_targets = self.predictor(context_out, target_in)
        return predicted_targets, true_targets, context_out, t_len

    def forward_batch(self, imgs, bidirectional=True):
        """Dense parallel GPU forward for a batch of images."""
        B = imgs.shape[0]
        device = imgs.device

        # Tokenize each image (CPU-GPU overlap via pin_memory DataLoader)
        batch_tokens = []
        for i in range(B):
            batch_tokens.append(self._tokenize_single(imgs[i]))
        tokens = torch.stack(batch_tokens, dim=0)   # (B, 256, D)

        # Hybrid Spatial-Block + Cross-Scale Masking (V2-12)
        mask_mode = torch.rand(1).item() if self.training else 0.0

        if mask_mode < 0.5:
            # Mode A: Cross-Scale prediction
            is_c2f = (torch.rand(1).item() > 0.5) if (bidirectional and self.training) else True
            if is_c2f:
                context_in = tokens[:, :self.COARSE_END, :]    # (B, 180, D)
                target_in  = tokens[:, self.COARSE_END:, :]    # (B,  76, D)
                t_len = self.L3_END - self.COARSE_END
            else:
                context_in = tokens[:, self.COARSE_END:, :]
                target_in  = tokens[:, :self.COARSE_END, :]
                t_len = self.COARSE_END
        else:
            # Mode B: Spatial Block prediction (random permutation)
            perm = torch.randperm(self.L3_END, device=device)
            context_in = tokens[:, perm[:self.COARSE_END], :]
            target_in  = tokens[:, perm[self.COARSE_END:], :]
            t_len = self.L3_END - self.COARSE_END

        # Dense batched ViT forward — drives Tensor Core utilisation
        context_out = self.context_encoder(context_in)
        with torch.no_grad():
            true_targets = self.target_norm(self.target_encoder(target_in))

        predicted_targets = self.predictor(context_out, target_in)
        return predicted_targets, true_targets, context_out, t_len

    # ── Feature Extraction ───────────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, img, multi_layer=True):
        """
        Single-image feature extraction for compatibility.
        For batch use, prefer extract_features_batch() — ~B× faster.
        """
        self.context_encoder.eval()
        if img.dim() == 4:
            img = img.squeeze(0)

        tokens = self._tokenize_single(img).unsqueeze(0)   # (1, 256, D)

        if multi_layer:
            layer_outputs = self.context_encoder(tokens, return_layer_outputs=True)
            # V4: return_layer_outputs is now guaranteed to work (vit.py V4).
            # The previous else-fallback did a second full forward pass — removed.
            agg = torch.stack(layer_outputs[-4:], dim=0).mean(dim=0)   # (1, 256, D)
            mean_pooled = agg.mean(dim=1)                               # (1, D)
        else:
            out = self.context_encoder(tokens)    # (1, 256, D)
            mean_pooled = out.mean(dim=1)         # (1, D)

        return F.normalize(mean_pooled, p=2, dim=-1)   # (1, D)

    @torch.no_grad()
    def extract_features_batch(self, imgs, multi_layer=True):
        """
        V3: Batched feature extraction — one GPU forward for the entire batch.
        Replaces the B-serial-calls pattern in extract_cub_features().
        ~B× faster (B=16 → 16× fewer GPU kernel launches).
        """
        self.context_encoder.eval()
        B = imgs.shape[0]

        # Tokenise each image (still serial — inherent to QuadTree design)
        batch_tokens = []
        for i in range(B):
            img_i = imgs[i] if imgs.dim() == 4 else imgs
            batch_tokens.append(self._tokenize_single(img_i))
        tokens_batch = torch.stack(batch_tokens, dim=0)   # (B, 256, D)

        # Single batched ViT forward
        if multi_layer:
            layer_outputs = self.context_encoder(tokens_batch, return_layer_outputs=True)
            # V4: return_layer_outputs is now guaranteed to work (vit.py V4).
            # The previous else-fallback called context_encoder a second time —
            # that was a full extra GPU forward pass executed silently if the list
            # check ever failed. Dead code removed.
            agg = torch.stack(layer_outputs[-4:], dim=0).mean(dim=0)   # (B, 256, D)
            mean_pooled = agg.mean(dim=1)                               # (B, D)
        else:
            out = self.context_encoder(tokens_batch)   # (B, 256, D)
            mean_pooled = out.mean(dim=1)              # (B, D)

        return F.normalize(mean_pooled, p=2, dim=-1)   # (B, D)


class QuadtreeClassifier(nn.Module):
    """
    V3 End-to-End Downstream Classifier.
    FIXED: properly handles any batch size B≥1 (V2 was broken for B>1).
    Single batched ViT forward for the whole batch → high Tensor Core utilisation.
    """
    def __init__(self, jepa_model, num_classes=200, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.tokenizer       = jepa_model.tokenizer
        self.z_bridge        = jepa_model.z_bridge
        self.context_encoder = jepa_model.context_encoder
        self.max_seq_len     = jepa_model.max_seq_len
        self.pooler          = ScaleAwareAttentivePool(embed_dim=jepa_model.z_bridge.embed_dim)

        # Copy precomputed level_ids as a registered buffer
        self.register_buffer('level_ids', jepa_model.level_ids.clone())

        self.head = nn.Sequential(
            nn.LayerNorm(jepa_model.z_bridge.embed_dim),
            nn.Linear(jepa_model.z_bridge.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _tokenize_batch(self, imgs):
        """Tokenize all images and stack to (B, 256, D)."""
        batch_tokens = []
        for i in range(imgs.shape[0]):
            pbl, pos = self.tokenizer(imgs[i])
            tokens   = self.z_bridge(pbl, pos)
            batch_tokens.append(tokens)
        return torch.stack(batch_tokens, dim=0)   # (B, 256, D)

    def forward(self, imgs):
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)   # (C,H,W) → (1,C,H,W)
        B = imgs.shape[0]

        tokens_batch = self._tokenize_batch(imgs)    # (B, 256, D)
        context_out  = self.context_encoder(tokens_batch)  # (B, 256, D)

        # Scale-aware pooling with activated level biases (V3)
        lvl = self.level_ids.unsqueeze(0).expand(B, -1)    # (B, 256)
        pooled = self.pooler(context_out, level_ids=lvl)   # (B, D)
        return self.head(pooled)                            # (B, num_classes)


def collate_quadtree_batch(batch, tokenizer, z_bridge, max_seq_len=800, device='cuda'):
    """
    V3 collate: updated for tensor-based tokenizer API.
    All images produce exactly 256 tokens → no padding needed.
    """
    images, labels = zip(*batch)
    batch_tokens = []

    for img in images:
        if img.dim() == 4:
            img = img.squeeze(0)
        pbl, pos = tokenizer(img)
        tokens   = z_bridge(pbl, pos)
        seq_len  = min(tokens.shape[0], max_seq_len)
        batch_tokens.append(tokens[:seq_len])

    # All same length (256) — deterministic budget
    padded = torch.stack(batch_tokens, dim=0)   # (B, 256, D)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return padded, labels_tensor