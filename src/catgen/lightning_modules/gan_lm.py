from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from catgen import artifacts
from catgen.models.dcgan import Discriminator, Generator


class GAN(L.LightningModule):
    def __init__(self, z_dim=128, ch=64, lr=2e-4, beta1=0.5, label_smooth=0.0, n_sample=64):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator(z_dim=z_dim, ch=ch)
        self.D = Discriminator(ch=ch)
        self.automatic_optimization = False
        self.register_buffer("fixed_z", torch.randn(n_sample, z_dim))
        self._d_losses = []
        self._g_losses = []
        self.run_dir: Path | None = None

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        return opt_g, opt_d

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        x = batch
        b = x.size(0)
        real_label = 1.0 - self.hparams.label_smooth
        fake_label = 0.0

        opt_d.zero_grad()
        d_real = self.D(x)
        loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, real_label))
        z = torch.randn(b, self.hparams.z_dim, device=self.device)
        fake = self.G(z).detach()
        d_fake = self.D(fake)
        loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.full_like(d_fake, fake_label))
        loss_d = loss_d_real + loss_d_fake
        self.manual_backward(loss_d)
        opt_d.step()

        opt_g.zero_grad()
        z = torch.randn(b, self.hparams.z_dim, device=self.device)
        fake = self.G(z)
        d_out = self.D(fake)
        loss_g = F.binary_cross_entropy_with_logits(d_out, torch.full_like(d_out, 1.0))
        self.manual_backward(loss_g)
        opt_g.step()

        self._d_losses.append(loss_d.item())
        self._g_losses.append(loss_g.item())

    def on_train_epoch_end(self):
        if self.run_dir is None:
            return
        d_avg = sum(self._d_losses) / max(len(self._d_losses), 1)
        g_avg = sum(self._g_losses) / max(len(self._g_losses), 1)
        self._d_losses.clear()
        self._g_losses.clear()

        with torch.no_grad():
            samples = self.G(self.fixed_z.to(self.device))
        sample_std = float(samples.std(dim=0).mean().item())

        artifacts.log_metrics(
            self.run_dir,
            step=self.global_step,
            epoch=self.current_epoch,
            split="train",
            values={"loss_d": d_avg, "loss_g": g_avg, "sample_std": sample_std},
        )
        artifacts.log_event(
            self.run_dir, "epoch_end",
            epoch=self.current_epoch, loss_d=d_avg, loss_g=g_avg, sample_std=sample_std,
        )
        self.log_dict({"loss_d": d_avg, "loss_g": g_avg, "sample_std": sample_std}, on_epoch=True, prog_bar=False)
        if sample_std < 0.05:
            artifacts.log_event(self.run_dir, "mode_collapse_warning", epoch=self.current_epoch, sample_std=sample_std)

        save_image(samples * 0.5 + 0.5,
                   self.run_dir / "samples" / f"epoch_{self.current_epoch:03d}.png",
                   nrow=8)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.hparams.z_dim, device=self.device)
        return self.G(z)
