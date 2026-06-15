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
        
    def get_2d_pos_embed(self, x, y, dim, device):
        half_dim = dim // 2
        omega = np.exp(np.arange(0, half_dim // 2, dtype=np.float32) * -(np.log(10000.0) / (half_dim // 2)))
        emb_x = x * omega
        emb_y = y * omega
        emb_x = np.concatenate([np.sin(emb_x), np.cos(emb_x)])
        emb_y = np.concatenate([np.sin(emb_y), np.cos(emb_y)])
        return torch.tensor(np.concatenate([emb_x, emb_y]), dtype=torch.float32, device=device).unsqueeze(0)

    def forward(self, patches, metadata):
        tokens = []
        for patch, meta in zip(patches, metadata):
            z_level = str(meta['Z'])
            flat_patch = patch.flatten().unsqueeze(0)
            projected_patch = self.projections[z_level](flat_patch)
            pe_2d = self.get_2d_pos_embed(meta['X'], meta['Y'], self.embed_dim, projected_patch.device)
            pe_1d = self.scale_embed(torch.tensor([meta['Z']], device=projected_patch.device))
            tokens.append(projected_patch + pe_2d + pe_1d)
        return torch.cat(tokens, dim=0)

class CrossAttentionPredictor(nn.Module):
    def __init__(self, embed_dim=768, heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, context_tokens, target_queries, context_mask=None, target_mask=None):
        attn_out, _ = self.cross_attn(
            query=target_queries, 
            key=context_tokens, 
            value=context_tokens,
            key_padding_mask=context_mask
        )
        x = self.norm1(target_queries + attn_out)
        x = self.norm2(x + self.mlp(x))
        if target_mask is not None:
            x = x * (~target_mask.unsqueeze(-1))
        return x

class QuadtreeJEPA(nn.Module):
    def __init__(self, base_vit, embed_dim=768, max_seq_len=800):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = QuadtreeTokenizer(thresholds=[0.28, 0.18, 0.11, 0.0])
        self.z_bridge = ZAxisFusionBridge(embed_dim=embed_dim)
        
        self.context_encoder = base_vit
        self.target_encoder = copy.deepcopy(base_vit)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = CrossAttentionPredictor(embed_dim=embed_dim)

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
            elif meta['Z'] == 3:
                target_tokens.append(token)
                
        if not context_tokens or not target_tokens:
            return None, None, None, None
            
        c_len = len(context_tokens)
        t_len = len(target_tokens)
        
        pad_context = torch.zeros(self.max_seq_len, tokens.size(-1), device=tokens.device)
        pad_target = torch.zeros(self.max_seq_len, tokens.size(-1), device=tokens.device)
        
        pad_context[:c_len] = torch.stack(context_tokens)
        pad_target[:t_len] = torch.stack(target_tokens)
        
        context_mask = torch.ones(self.max_seq_len, dtype=torch.bool, device=tokens.device)
        context_mask[:c_len] = False
        
        target_mask = torch.ones(self.max_seq_len, dtype=torch.bool, device=tokens.device)
        target_mask[:t_len] = False
        
        context_in = pad_context.unsqueeze(0)
        target_in = pad_target.unsqueeze(0)
        
        context_out = self.context_encoder(context_in)
        
        with torch.no_grad():
            true_targets = self.target_encoder(target_in)
            
        predicted_targets = self.predictor(
            context_out, 
            target_in, 
            context_mask=context_mask.unsqueeze(0), 
            target_mask=target_mask
        )
        
        return predicted_targets, true_targets, target_mask, t_len