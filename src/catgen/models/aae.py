from __future__ import annotations

import math

import torch
import torch.nn as nn


def _stages(image_size: int) -> int:
    if image_size < 4 or image_size & (image_size - 1):
        raise ValueError(f"image_size must be power of 2 >= 4, got {image_size}")
    return int(math.log2(image_size // 4))


class Encoder(nn.Module):
    def __init__(self, z_dim=128, ch=64, image_size=64):
        super().__init__()
        n_down = _stages(image_size)
        layers: list[nn.Module] = [
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ]
        cin, cout = ch, ch * 2
        for i in range(1, n_down):
            layers += [
                nn.Conv2d(cin, cout, 4, 2, 1),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, True),
            ]
            cin, cout = cout, min(cout * 2, ch * 8)
        layers.append(nn.Conv2d(cin, z_dim, 4, 1, 0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


class Decoder(nn.Module):
    def __init__(self, z_dim=128, ch=64, image_size=64):
        super().__init__()
        self.z_dim = z_dim
        stages = _stages(image_size)
        ch_schedule = [ch * 4, ch * 2, ch] + [ch] * max(0, stages - 3)
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(z_dim, ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),
        ]
        cin = ch * 8
        for i in range(stages):
            cout = 3 if i == stages - 1 else ch_schedule[min(i, len(ch_schedule) - 1)]
            layers.append(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False))
            if cout != 3:
                layers += [nn.BatchNorm2d(cout), nn.ReLU(True)]
            else:
                layers.append(nn.Tanh())
            cin = cout
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class LatentDiscriminator(nn.Module):
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).view(-1)
