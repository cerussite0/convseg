import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext_block import DropPath

class SpatialReductionAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, sr_ratio: int=1, qkv_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        assert dim % num_heads == 0, f'dim ({dim}) must be divisible by num_heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_kv = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_kv = self.sr_conv(x_kv)
            x_kv = x_kv.flatten(2).permute(0, 2, 1)
            x_kv = self.sr_norm(x_kv)
        else:
            x_kv = x
        N_kv = x_kv.shape[1]
        k = self.k(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MixFFN(nn.Module):

    def __init__(self, dim: int, ffn_expansion: int=4, drop: float=0.0):
        super().__init__()
        hidden = dim * ffn_expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, sr_ratio: int=1, ffn_expansion: int=4, drop: float=0.0, drop_path: float=0.0, qkv_bias: bool=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim=dim, ffn_expansion=ffn_expansion, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x