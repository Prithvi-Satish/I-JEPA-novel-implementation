import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class QuadtreeTokenizer(nn.Module):
    def __init__(self, thresholds):
        super().__init__()
        self.thresholds = thresholds
        self.patch_sizes = {0: 64, 1: 32, 2: 16, 3: 8}

    def compute_variance(self, patch):
        if patch.shape[0] == 3:
            gray = 0.2989 * patch[0] + 0.5870 * patch[1] + 0.1140 * patch[2]
            return gray.var()
        return patch.var()

    def split_patch(self, image, x, y, size, level, patches, metadata):
        patch = image[:, y:y+size, x:x+size]
        if level == 3 or self.compute_variance(patch) < self.thresholds[level]:
            patches.append(patch)
            metadata.append({'X': x, 'Y': y, 'Z': level})
        else:
            new_size = size // 2
            new_level = level + 1
            for dy in [0, new_size]:
                for dx in [0, new_size]:
                    self.split_patch(image, x + dx, y + dy, new_size, new_level, patches, metadata)

    def forward(self, image):
        if image.dim() == 4:
            image = image.squeeze(0)
        C, H, W = image.shape
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        _, pad_H, pad_W = image.shape
        patches = []
        metadata = []
        size = self.patch_sizes[0]
        
        for y in range(0, pad_H, size):
            for x in range(0, pad_W, size):
                self.split_patch(image, x, y, size, 0, patches, metadata)
                
        if len(metadata) > 800:
            patches = patches[:800]
            metadata = metadata[:800]
            
        return patches, metadata

class ZAxisFusionBridge(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict({
            '0': nn.Linear(3 * 64 * 64, embed_dim),
            '1': nn.Linear(3 * 32 * 32, embed_dim),
            '2': nn.Linear(3 * 16 * 16, embed_dim),
            '3': nn.Linear(3 * 8 * 8, embed_dim)
        })
        self.scale_embed = nn.Embedding(4, embed_dim)
        
        half_dim = embed_dim // 2
        omega = torch.exp(torch.arange(0, half_dim // 2, dtype=torch.float32) * -(np.log(10000.0) / (half_dim // 2)))
        self.register_buffer('omega', omega)

    def get_2d_pos_embed_tensor(self, xs, ys, device):
        omega = self.omega.to(device)
        emb_x = xs * omega.unsqueeze(0)
        emb_y = ys * omega.unsqueeze(0)
        emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=-1)
        emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=-1)
        return torch.cat([emb_x, emb_y], dim=-1)

    def forward(self, patches, metadata):
        if not patches:
            device = next(self.parameters()).device
            return torch.empty(0, self.embed_dim, device=device)
            
        device = next(self.parameters()).device
        N = len(patches)
        
        # High-performance grouped projection: 4 batched matrix multiplies instead of N individual kernel calls
        level_groups = {'0': [], '1': [], '2': [], '3': []}
        level_indices = {'0': [], '1': [], '2': [], '3': []}
        
        for idx, (patch, meta) in enumerate(zip(patches, metadata)):
            z_level = str(meta['Z'])
            level_groups[z_level].append(patch.flatten())
            level_indices[z_level].append(idx)
            
        projected = None
        for z_level, p_list in level_groups.items():
            if p_list:
                stacked = torch.stack(p_list).to(device)
                proj = self.projections[z_level](stacked)
                if projected is None:
                    projected = torch.empty(N, self.embed_dim, device=device, dtype=proj.dtype)
                indices = torch.tensor(level_indices[z_level], device=device)
                projected[indices] = proj
        
        if projected is None:
            return torch.empty(0, self.embed_dim, device=device)
            
        xs = torch.tensor([m['X'] for m in metadata], dtype=torch.float32, device=device).unsqueeze(1)
        ys = torch.tensor([m['Y'] for m in metadata], dtype=torch.float32, device=device).unsqueeze(1)
        zs = torch.tensor([m['Z'] for m in metadata], dtype=torch.long, device=device)
        
        pe_2d = self.get_2d_pos_embed_tensor(xs, ys, device).to(projected.dtype)
        pe_1d = self.scale_embed(zs).to(projected.dtype)
        
        return projected + pe_2d + pe_1d

class PredictorBlock(nn.Module):
    def __init__(self, embed_dim=768, heads=8, is_cross=False):
        super().__init__()
        self.is_cross = is_cross
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None, context_mask=None):
        if self.is_cross and context is not None:
            attn_out, _ = self.attn(query=x, key=context, value=context, key_padding_mask=context_mask)
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
        # Layer 1: Cross-Attention (Target queries gather information from context)
        x = self.layers[0](target_queries, context=context_tokens, context_mask=context_mask)
        # Layers 2..depth: Self-Attention (Target tokens refine spatial predictions among themselves)
        for layer in self.layers[1:]:
            x = layer(x)
        x = self.final_norm(x)
        if target_mask is not None:
            x = x * (~target_mask.unsqueeze(-1))
        return x

class QuadtreeJEPA(nn.Module):
    def __init__(self, base_vit, embed_dim=768, max_seq_len=800, predictor_depth=3):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = QuadtreeTokenizer(thresholds=[0.28, 0.18, 0.11, 0.0])
        self.z_bridge = ZAxisFusionBridge(embed_dim=embed_dim)
        
        self.context_encoder = base_vit
        self.target_encoder = copy.deepcopy(base_vit)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = CrossAttentionPredictor(embed_dim=embed_dim, depth=predictor_depth)

    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def forward(self, img):
        patches, metadata = self.tokenizer(img)
        if not metadata:
            return None, None, None, None
            
        tokens = self.z_bridge(patches, metadata)
        
        context_tokens = []
        target_tokens = []
        
        for token, meta in zip(tokens, metadata):
            if meta['Z'] in [0, 1]:
                context_tokens.append(token)
            elif meta['Z'] in [2, 3]:
                # Both Level 2 (16x16) and Level 3 (8x8) serve as fine-grained target queries
                target_tokens.append(token)
                
        # Adaptive partition fallback: if an image is smooth (e.g. healthy leaves with only Level 0/1 tokens)
        # or has no coarse context, partition the available tokens (70% context, 30% target)
        # so 100% of all images contribute to representation learning.
        if not context_tokens or not target_tokens:
            if len(tokens) < 2:
                return None, None, None, None
            n_ctx = max(1, int(len(tokens) * 0.70))
            context_tokens = [tokens[i] for i in range(n_ctx)]
            target_tokens = [tokens[i] for i in range(n_ctx, len(tokens))]
            if not target_tokens:
                target_tokens = [tokens[-1]]
            
        c_len = min(len(context_tokens), self.max_seq_len)
        t_len = min(len(target_tokens), self.max_seq_len)
        
        # Dynamic unpadded sequence tensors: eliminates 800-token zero padding & cuts attention FLOPs drastically
        context_in = torch.stack(context_tokens[:c_len]).unsqueeze(0)
        target_in = torch.stack(target_tokens[:t_len]).unsqueeze(0)
        
        context_out = self.context_encoder(context_in)
        
        with torch.no_grad():
            true_targets = self.target_encoder(target_in)
            
        predicted_targets = self.predictor(
            context_out, 
            target_in
        )
        
        return predicted_targets, true_targets, None, t_len

    @torch.no_grad()
    def extract_features(self, img):
        """
        Extracts a multi-scale representation vector (1, embed_dim) for downstream evaluation.
        Passes all valid quadtree tokens (Levels 0, 1, 2, and 3) dynamically through the context encoder.
        """
        self.context_encoder.eval()
        patches, metadata = self.tokenizer(img)
        if not metadata:
            return torch.zeros(1, self.z_bridge.embed_dim, device=img.device)
            
        tokens = self.z_bridge(patches, metadata)
        if len(tokens) == 0:
            return torch.zeros(1, self.z_bridge.embed_dim, device=img.device)
            
        seq_len = min(len(tokens), self.max_seq_len)
        tokens_in = tokens[:seq_len].unsqueeze(0)
        
        context_out = self.context_encoder(tokens_in)
        valid_features = context_out[0]
        return valid_features.mean(dim=0, keepdim=True)


class QuadtreeClassifier(nn.Module):
    """
    End-to-end downstream classifier wrapper for Quadtree-JEPA.
    Processes dynamic multi-scale tokens through the context encoder and projects to class logits.
    Supports discriminative fine-tuning (unfrozen backbone) or frozen feature probing.
    """
    def __init__(self, jepa_model, num_classes=18):
        super().__init__()
        self.tokenizer = jepa_model.tokenizer
        self.z_bridge = jepa_model.z_bridge
        self.context_encoder = jepa_model.context_encoder
        self.max_seq_len = jepa_model.max_seq_len
        self.head = nn.Linear(jepa_model.z_bridge.embed_dim, num_classes)

    def forward(self, img):
        if img.dim() == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
            
        patches, metadata = self.tokenizer(img)
        if not metadata:
            return torch.zeros(1, self.head.out_features, device=img.device)
            
        tokens = self.z_bridge(patches, metadata)
        if len(tokens) == 0:
            return torch.zeros(1, self.head.out_features, device=img.device)
            
        seq_len = min(len(tokens), self.max_seq_len)
        tokens_in = tokens[:seq_len].unsqueeze(0)
        
        context_out = self.context_encoder(tokens_in)
        valid_features = context_out[0]
        pooled = valid_features.mean(dim=0, keepdim=True)
        return self.head(pooled)