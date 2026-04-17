# models/discriminator.py

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import NUM_ATTRIBUTES, CHANNELS, FEATURES_D, IMAGE_SIZE


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    Conditional DCGAN Discriminator (Critic in WGAN terminology).

    Input:
        image : [batch, 3, 64, 64]
        attrs : [batch, NUM_ATTRIBUTES]

    Output:
        score : [batch, 1]
    """
    def __init__(self):
        super().__init__()

        fd = FEATURES_D  # 64

        # Project attribute vector → [batch, 64*64]
        self.attr_embed = nn.Sequential(
            nn.Linear(NUM_ATTRIBUTES, 64 * 64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        in_channels = CHANNELS + 1

        self.block1 = DiscriminatorBlock(in_channels, fd,      normalize=False)
        self.block2 = DiscriminatorBlock(fd,          fd * 2)
        self.block3 = DiscriminatorBlock(fd * 2,      fd * 4)
        self.block4 = DiscriminatorBlock(fd * 4,      fd * 8)

        self.output = nn.Conv2d(fd * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self._initialize_weights()

    def forward(self, image, attrs):
        batch = image.size(0)

        attr_map = self.attr_embed(attrs.float())        # [batch, 4096]
        attr_map = attr_map.view(batch, 1, 64, 64)       # [batch, 1, 64, 64]

        x = torch.cat([image, attr_map], dim=1)          # [batch, 4, 64, 64]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)

        return x.view(batch, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import DEVICE

    D = Discriminator().to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in D.parameters()):,}")

    images = torch.randn(4, CHANNELS, 64, 64).to(DEVICE)
    attrs  = torch.randint(0, 2, (4, NUM_ATTRIBUTES)).float().to(DEVICE)
    scores = D(images, attrs)

    print(f"Input  image  : {images.shape}")
    print(f"Input  attrs  : {attrs.shape}")
    print(f"Output scores : {scores.shape}")
