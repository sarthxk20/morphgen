# models/generator.py

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LATENT_DIM, NUM_ATTRIBUTES, CHANNELS, FEATURES_G


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Conditional DCGAN Generator.

    Input:
        z    : [batch, LATENT_DIM]
        attrs: [batch, NUM_ATTRIBUTES]

    Output:
        image: [batch, 3, 64, 64]

    Architecture:
        4x4 → 8x8 → 16x16 → 32x32 → 64x64
    """
    def __init__(self):
        super().__init__()

        self.input_dim = LATENT_DIM + NUM_ATTRIBUTES  # 128 + 40 = 168
        fg = FEATURES_G  # 64

        # Project: (168,) → (fg*8, 4, 4)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, fg * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fg * 8),
            nn.ReLU(inplace=True)
        )
        # 4x4 → 8x8
        self.block1 = GeneratorBlock(fg * 8, fg * 4)
        # 8x8 → 16x16
        self.block2 = GeneratorBlock(fg * 4, fg * 2)
        # 16x16 → 32x32
        self.block3 = GeneratorBlock(fg * 2, fg)

        # 32x32 → 64x64 — final layer, no BatchNorm, Tanh output
        self.output = nn.Sequential(
            nn.ConvTranspose2d(fg, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, z, attrs):
        x = torch.cat([z, attrs.float()], dim=1)       # [batch, 168]
        x = x.unsqueeze(-1).unsqueeze(-1)               # [batch, 168, 1, 1]
        x = self.project(x)   # [batch, fg*8,  4,  4]
        x = self.block1(x)    # [batch, fg*4,  8,  8]
        x = self.block2(x)    # [batch, fg*2,  16, 16]
        x = self.block3(x)    # [batch, fg,    32, 32]
        x = self.output(x)    # [batch, 3,     64, 64]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import DEVICE

    G = Generator().to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in G.parameters()):,}")

    z     = torch.randn(4, LATENT_DIM).to(DEVICE)
    attrs = torch.randint(0, 2, (4, NUM_ATTRIBUTES)).to(DEVICE)
    out   = G(z, attrs)
    print(f"Output shape: {out.shape}")   # expect [4, 3, 64, 64]
    print(f"Pixel range : [{out.min():.2f}, {out.max():.2f}]")
