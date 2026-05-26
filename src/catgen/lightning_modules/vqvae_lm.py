from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from catgen import artifacts
from catgen.models.vqvae import VQVAE


class VQ(L.LightningModule):
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden=128, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = VQVAE(num_embeddings, embedding_dim, hidden)
        self._buf = {"recon": [], "commit": [], "perp": []}
        self._code_counts = torch.zeros(num_embeddings, dtype=torch.long)
        self.run_dir: Path | None = None
        self._last_sample_batch = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x = batch
        x_rec, commit, idx, perp = self.net(x)
        loss_recon = F.mse_loss(x_rec, x)
        loss = loss_recon + commit
        self._buf["recon"].append(loss_recon.item())
        self._buf["commit"].append(commit.item())
        self._buf["perp"].append(perp.item())
        self._code_counts.scatter_add_(
            0, idx.flatten().detach().cpu(), torch.ones_like(idx.flatten().detach().cpu())
        )
        self._last_sample_batch = x.detach()
        return loss

    def on_train_epoch_end(self):
        if self.run_dir is None:
            return
        avgs = {k: sum(v) / max(len(v), 1) for k, v in self._buf.items()}
        for v in self._buf.values():
            v.clear()
        dead = int((self._code_counts == 0).sum().item())
        used = int((self._code_counts > 0).sum().item())

        artifacts.log_metrics(
            self.run_dir,
            step=self.global_step, epoch=self.current_epoch, split="train",
            values={**avgs, "codes_used": used, "codes_dead": dead},
        )
        artifacts.log_event(self.run_dir, "epoch_end", epoch=self.current_epoch, **avgs, codes_used=used, codes_dead=dead)
        self.log_dict({**avgs, "codes_used": float(used), "codes_dead": float(dead)}, on_epoch=True, prog_bar=False)
        if dead > self.hparams.num_embeddings * 0.5:
            artifacts.log_event(self.run_dir, "codebook_dead_codes", epoch=self.current_epoch, dead=dead)

        if self._last_sample_batch is not None:
            x = self._last_sample_batch[:16].to(self.device)
            with torch.no_grad():
                x_rec, *_ = self.net(x)
            grid = torch.cat([x, x_rec], dim=0)
            save_image(grid * 0.5 + 0.5,
                       self.run_dir / "samples" / f"epoch_{self.current_epoch:03d}.png",
                       nrow=16)

        np.save(self.run_dir / "eval" / "codebook_counts.npy", self._code_counts.numpy())

    def on_train_end(self):
        if self.run_dir is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            counts = self._code_counts.numpy()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(len(counts)), counts, width=1.0)
            ax.set_xlabel("code id")
            ax.set_ylabel("uses (cumulative)")
            ax.set_title(f"codebook usage (K={len(counts)}, dead={int((counts == 0).sum())})")
            fig.tight_layout()
            fig.savefig(self.run_dir / "eval" / "codebook_hist.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            artifacts.log_event(self.run_dir, "codebook_hist_failed", error=repr(e))

    @torch.no_grad()
    def empirical_code_freq(self, loader, device) -> torch.Tensor:
        counts = torch.zeros(self.hparams.num_embeddings, dtype=torch.long)
        for x in loader:
            x = x.to(device)
            z_e = self.net.enc(x)
            _, _, idx, _ = self.net.vq(z_e)
            counts.scatter_add_(0, idx.flatten().cpu(), torch.ones_like(idx.flatten().cpu()))
        return counts.float() / counts.sum().clamp(min=1)

    @torch.no_grad()
    def sample_from_freq(self, n: int, freq: torch.Tensor, grid: int, device) -> torch.Tensor:
        probs = freq.to(device)
        idx = torch.multinomial(probs, n * grid * grid, replacement=True).view(n, grid, grid)
        z_q = self.net.vq.embedding[idx].permute(0, 3, 1, 2).contiguous()
        return self.net.dec(z_q)
