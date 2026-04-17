import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ConvSegEncoder
from .decoder import ConvSegDecoder

class ConvSegNet(nn.Module):

    def __init__(self, num_classes: int=19, channels: list[int]=(64, 128, 320, 512), depths: list[int]=(3, 3, 6, 3), sr_ratios: list[int]=(8, 4, 2, 1), num_heads: list[int]=(1, 2, 5, 8), embed_dim: int=256, drop_path_rate: float=0.1, drop_rate: float=0.0, decoder_dropout: float=0.1, in_channels: int=3, gate_sr_ratio: int=1):
        super().__init__()
        self.encoder = ConvSegEncoder(channels=channels, depths=depths, sr_ratios=sr_ratios, num_heads=num_heads, drop_path_rate=drop_path_rate, drop_rate=drop_rate, in_channels=in_channels)
        self.decoder = ConvSegDecoder(in_channels_list=list(channels), embed_dim=embed_dim, num_classes=num_classes, dropout=decoder_dropout, gate_sr_ratio=gate_sr_ratio)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits