import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjection(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class CrossAttentionBoundaryGate(nn.Module):

    def __init__(self, embed_dim: int=256, num_heads: int=8, attn_drop: float=0.0, sr_ratio: int=1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, f4_hat: torch.Tensor, f1_hat: torch.Tensor, f2_hat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f4_hat.shape
        N = H * W
        q_in = f4_hat.flatten(2).permute(0, 2, 1)
        boundary = torch.cat([f1_hat, f2_hat], dim=2)
        if self.sr_ratio > 1:
            boundary = self.sr_conv(boundary)
            kv_in = boundary.flatten(2).permute(0, 2, 1)
            kv_in = self.sr_norm(kv_in)
        else:
            kv_in = boundary.flatten(2).permute(0, 2, 1)
        N_kv = kv_in.shape[1]
        q = self.q_proj(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        g_raw = (attn @ v).transpose(1, 2).reshape(B, N, C)
        g_raw = self.out_proj(g_raw)
        sigma = torch.sigmoid(self.gate_proj(torch.cat([q_in, g_raw], dim=-1)))
        g = sigma * g_raw + (1.0 - sigma) * q_in
        g = g.permute(0, 2, 1).reshape(B, C, H, W)
        return g

class ConvSegDecoder(nn.Module):

    def __init__(self, in_channels_list: list[int]=(64, 128, 320, 512), embed_dim: int=256, num_classes: int=19, dropout: float=0.1, gate_num_heads: int=8, gate_sr_ratio: int=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_projs = nn.ModuleList([MLPProjection(c, embed_dim) for c in in_channels_list])
        self.boundary_gate = CrossAttentionBoundaryGate(embed_dim=embed_dim, num_heads=gate_num_heads, sr_ratio=gate_sr_ratio)
        self.fusion = nn.Sequential(nn.Conv2d(4 * embed_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        assert len(features) == 4
        projected = [proj(feat) for proj, feat in zip(self.mlp_projs, features)]
        target_size = projected[0].shape[2:]
        upsampled = []
        for i, p in enumerate(projected):
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(p)
        f1_hat, f2_hat, f3_hat, f4_hat = upsampled
        g = self.boundary_gate(f4_hat, f1_hat, f2_hat)
        fused = torch.cat([f1_hat, f2_hat, f3_hat, g], dim=1)
        out = self.fusion(fused)
        out = self.dropout(out)
        out = self.cls_head(out)
        return out