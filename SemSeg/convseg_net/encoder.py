import torch
import torch.nn as nn
from .convnext_block import ConvNeXtBlock, LayerNormChannels
from .transformer_block import TransformerBlock

class PatchEmbedStem(nn.Module):

    def __init__(self, in_channels: int=3, embed_dim: int=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm = LayerNormChannels(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))

class DownsampleLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = LayerNormChannels(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))

class ConvSegEncoder(nn.Module):

    def __init__(self, channels: list[int]=(64, 128, 320, 512), depths: list[int]=(3, 3, 6, 3), sr_ratios: list[int]=(8, 4, 2, 1), num_heads: list[int]=(1, 2, 5, 8), drop_path_rate: float=0.1, drop_rate: float=0.0, qkv_bias: bool=True, in_channels: int=3):
        super().__init__()
        self.num_stages = 4
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0
        self.stem = PatchEmbedStem(in_channels, channels[0])
        stage1_blocks = []
        for i in range(depths[0]):
            stage1_blocks.append(ConvNeXtBlock(channels[0], drop_path=dpr[block_idx]))
            block_idx += 1
        self.stage1 = nn.Sequential(*stage1_blocks)
        self.down1 = DownsampleLayer(channels[0], channels[1])
        stage2_blocks = []
        for i in range(depths[1]):
            stage2_blocks.append(ConvNeXtBlock(channels[1], drop_path=dpr[block_idx]))
            block_idx += 1
        self.stage2 = nn.Sequential(*stage2_blocks)
        self.down2 = DownsampleLayer(channels[1], channels[2])
        stage3_blocks = []
        for i in range(depths[2]):
            stage3_blocks.append(TransformerBlock(dim=channels[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2], drop=drop_rate, drop_path=dpr[block_idx], qkv_bias=qkv_bias))
            block_idx += 1
        self.stage3 = nn.ModuleList(stage3_blocks)
        self.norm3 = nn.LayerNorm(channels[2])
        self.down3 = DownsampleLayer(channels[2], channels[3])
        stage4_blocks = []
        for i in range(depths[3]):
            stage4_blocks.append(TransformerBlock(dim=channels[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3], drop=drop_rate, drop_path=dpr[block_idx], qkv_bias=qkv_bias))
            block_idx += 1
        self.stage4 = nn.ModuleList(stage4_blocks)
        self.norm4 = nn.LayerNorm(channels[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.stem(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.down1(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.down2(x)
        B, C, H3, W3 = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        for blk in self.stage3:
            x = blk(x, H3, W3)
        x = self.norm3(x)
        x = x.permute(0, 2, 1).reshape(B, C, H3, W3)
        outs.append(x)
        x = self.down3(x)
        B, C, H4, W4 = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        for blk in self.stage4:
            x = blk(x, H4, W4)
        x = self.norm4(x)
        x = x.permute(0, 2, 1).reshape(B, C, H4, W4)
        outs.append(x)
        return outs