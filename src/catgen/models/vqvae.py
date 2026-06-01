from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, decay=0.99, eps=1e-5, commitment=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        emb = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer("embedding", emb)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", emb.clone())

    def forward(self, z_e):
        b, c, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, c)
        d = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        idx = d.argmin(1)
        onehot = F.one_hot(idx, self.K).type(flat.dtype)
        z_q = (onehot @ self.embedding).view(b, h, w, c).permute(0, 3, 1, 2)

        if self.training:
            self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
            dw = onehot.t() @ flat
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
            self.embedding.copy_(self.ema_w / cs.unsqueeze(1))

        commit = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return z_q_st, commit * self.commitment, idx.view(b, h, w), perplexity


def _vq_down_stages(image_size: int) -> int:
    """Stride-2 stages until spatial size is image_size/4 (16 at 64, 16 at 128)."""
    return int(math.log2(image_size // 16))


class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=128, emb_dim=64, image_size=64):
        super().__init__()
        n_down = _vq_down_stages(image_size)
        layers: list[nn.Module] = []
        cin = in_ch
        for i in range(n_down):
            cout = hidden // 2 if i == 0 else hidden
            layers += [nn.Conv2d(cin, cout, 4, 2, 1), nn.ReLU(True)]
            cin = cout
        layers += [
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, emb_dim, 1, 1, 0),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, emb_dim=64, hidden=128, out_ch=3, image_size=64):
        super().__init__()
        n_up = _vq_down_stages(image_size)
        layers: list[nn.Module] = [
            nn.Conv2d(emb_dim, hidden, 3, 1, 1),
            nn.ReLU(True),
        ]
        cin = hidden
        for i in range(n_up):
            cout = out_ch if i == n_up - 1 else (hidden // 2 if i == n_up - 2 else hidden)
            if cout == out_ch:
                layers += [nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.Tanh()]
            else:
                layers += [nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.ReLU(True)]
            cin = cout
        self.net = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.net(z_q)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden=128, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.enc = Encoder(hidden=hidden, emb_dim=embedding_dim, image_size=image_size)
        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim)
        self.dec = Decoder(emb_dim=embedding_dim, hidden=hidden, image_size=image_size)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, commit, idx, perp = self.vq(z_e)
        x_rec = self.dec(z_q)
        return x_rec, commit, idx, perp
