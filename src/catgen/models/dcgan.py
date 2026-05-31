from __future__ import annotations

import torch
import torch.nn as nn


def _up_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(True),
    )


class GeneratorUpsample(nn.Module):
    """Upsample + conv generator (fewer checkerboard artifacts than ConvTranspose2d)."""

    def __init__(self, z_dim=128, ch=64):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, ch * 8 * 4 * 4)
        self.blocks = nn.Sequential(
            _up_block(ch * 8, ch * 4),
            _up_block(ch * 4, ch * 2),
            _up_block(ch * 2, ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        b = z.size(0)
        x = self.fc(z).view(b, -1, 4, 4)
        return self.blocks(x)


class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)
