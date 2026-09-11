import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ─────────────────────────────────────────────────────────────────────────────
#  DropPath (Stochastic Depth) — V4 addition
#  Randomly drops entire residual branches during training.
#  At test/eval time (self.training=False) it's a pure identity — zero cost.
#
#  Why DropPath and NOT standard Dropout:
#    Dropout zeroes individual activations → partial information remains.
#    DropPath zeroes the ENTIRE residual contribution of one block → forces
#    the network to route information through other layers → implicit ensemble
#    of models of varying depth → strongest regularizer for small datasets.
#
#  NOTE: target_encoder is set to eval() during SSL → DropPath inactive on
#  target → targets remain deterministic → NO conflict with JEPA SSL signal.
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(Module):
    """Stochastic depth — drop entire residual path with probability drop_prob."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob   = 1.0 - self.drop_prob
        # shape: (B, 1, 1, ...) — same as x but all spatial/token dims = 1
        shape       = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        # Bernoulli mask — scale up to keep expected magnitude
        output = x * (random_tensor < keep_prob).float() / keep_prob
        return output


class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0., drop_path=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., drop_path=0., qkv_bias=True):
        super().__init__()
        inner_dim    = dim_head * heads
        project_out  = not (heads == 1 and dim_head == dim)

        self.heads   = heads
        self.scale   = dim_head ** -0.5

        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x   = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        drop_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.drop_path(self.to_out(out))


class Transformer(Module):
    """
    ViT Transformer with per-layer DropPath (stochastic depth).

    drop_path_rate: total drop rate for the DEEPEST layer.
    Intermediate layers use linear interpolation from 0 → drop_path_rate.
    This is the standard DeiT/DINO stochastic depth schedule.

    return_layer_outputs=True: returns a list of (B, N, D) tensors,
    one per layer (after LayerNorm), for multi-layer feature readout.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., drop_path_rate=0.0, qkv_bias=True):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        # Linear stochastic depth schedule: 0 → drop_path_rate
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        for i in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head,
                          dropout=dropout, drop_path=dpr[i], qkv_bias=qkv_bias),
                FeedForward(dim, mlp_dim, dropout=dropout, drop_path=dpr[i])
            ]))

    def forward(self, x, return_layer_outputs=False):
        outputs = []
        for attn, ff in self.layers:
            x = attn(x) + x      # DropPath inside attn.forward()
            x = ff(x) + x        # DropPath inside ff.forward()
            if return_layer_outputs:
                outputs.append(self.norm(x).clone())

        if return_layer_outputs:
            return outputs        # list of (B, N, D), length = depth
        return self.norm(x)


class ViT(Module):
    """
    Vision Transformer backbone — V5 (Pre-trained support).
    Added: DropPath (stochastic depth), return_layer_outputs, qkv_bias.

    Parameters
    ----------
    dim             : token embedding dimension (768)
    depth           : number of transformer layers (12)
    heads           : number of attention heads (12)
    mlp_dim         : FFN hidden dimension (3072 standard)
    dim_head        : per-head dimension (64), heads*dim_head should == dim
    dropout         : attention + FFN dropout (0.1)
    emb_dropout     : input token dropout (0.0)
    drop_path_rate  : stochastic depth max rate (0.1 recommended for depth=12)
    qkv_bias        : whether QKV projections have bias (True for ImageNet ViTs)
    """
    def __init__(self, *, dim=768, depth=12, heads=12, mlp_dim=3072,
                 dim_head=64, dropout=0., emb_dropout=0., drop_path_rate=0.0,
                 qkv_bias=True):
        super().__init__()
        self.dropout     = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout=dropout,
                                       drop_path_rate=drop_path_rate,
                                       qkv_bias=qkv_bias)

    def forward(self, tokens, return_layer_outputs=False):
        x = self.dropout(tokens)
        return self.transformer(x, return_layer_outputs=return_layer_outputs)


def load_pretrained_vit_weights(vit_model, model_name='vit_base_patch16_224'):
    """
    Loads ImageNet pre-trained Transformer weights from timm into ViT.
    Transfers 100% of the 12 Transformer blocks (QKV, proj, LayerNorm, MLP) and final norm.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required to load pre-trained ViT weights. Run: pip install timm")

    print(f"Loading pre-trained weights from '{model_name}'...")
    src_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    src_sd = src_model.state_dict()
    dst_sd = vit_model.state_dict()

    new_sd = {}
    depth = len(vit_model.transformer.layers)
    for i in range(depth):
        # LayerNorm 1
        new_sd[f'transformer.layers.{i}.0.norm.weight'] = src_sd[f'blocks.{i}.norm1.weight']
        new_sd[f'transformer.layers.{i}.0.norm.bias']   = src_sd[f'blocks.{i}.norm1.bias']
        # QKV Attention
        new_sd[f'transformer.layers.{i}.0.to_qkv.weight'] = src_sd[f'blocks.{i}.attn.qkv.weight']
        if f'transformer.layers.{i}.0.to_qkv.bias' in dst_sd:
            new_sd[f'transformer.layers.{i}.0.to_qkv.bias'] = src_sd[f'blocks.{i}.attn.qkv.bias']
        # Attention Projection
        new_sd[f'transformer.layers.{i}.0.to_out.0.weight'] = src_sd[f'blocks.{i}.attn.proj.weight']
        new_sd[f'transformer.layers.{i}.0.to_out.0.bias']   = src_sd[f'blocks.{i}.attn.proj.bias']
        # LayerNorm 2
        new_sd[f'transformer.layers.{i}.1.net.0.weight'] = src_sd[f'blocks.{i}.norm2.weight']
        new_sd[f'transformer.layers.{i}.1.net.0.bias']   = src_sd[f'blocks.{i}.norm2.bias']
        # MLP FC1
        new_sd[f'transformer.layers.{i}.1.net.1.weight'] = src_sd[f'blocks.{i}.mlp.fc1.weight']
        new_sd[f'transformer.layers.{i}.1.net.1.bias']   = src_sd[f'blocks.{i}.mlp.fc1.bias']
        # MLP FC2
        new_sd[f'transformer.layers.{i}.1.net.4.weight'] = src_sd[f'blocks.{i}.mlp.fc2.weight']
        new_sd[f'transformer.layers.{i}.1.net.4.bias']   = src_sd[f'blocks.{i}.mlp.fc2.bias']

    # Final LayerNorm
    new_sd['transformer.norm.weight'] = src_sd['norm.weight']
    new_sd['transformer.norm.bias']   = src_sd['norm.bias']

    msg = vit_model.load_state_dict(new_sd, strict=False)
    print(f"--> Successfully transferred {len(new_sd)} pre-trained weight tensors into ViT backbone!")
    return vit_model

