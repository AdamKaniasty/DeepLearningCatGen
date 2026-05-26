from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from catgen import artifacts
from catgen.models.aae import Decoder, Encoder, LatentDiscriminator


class AAE(L.LightningModule):
    def __init__(self, z_dim=128, ch=64, lr=1e-4, recon_w=1.0, adv_w=1.0, n_sample=64):
        super().__init__()
        self.save_hyperparameters()
        self.enc = Encoder(z_dim=z_dim, ch=ch)
        self.dec = Decoder(z_dim=z_dim, ch=ch)
        self.disc = LatentDiscriminator(z_dim=z_dim)
        self.automatic_optimization = False
        self.register_buffer("fixed_z", torch.randn(n_sample, z_dim))
        self._buf = {"recon": [], "g_adv": [], "d_adv": []}
        self._last_batch = None
        self.run_dir: Path | None = None

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.hparams.lr)
        opt_g = torch.optim.Adam(self.enc.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.hparams.lr)
        return opt_ae, opt_g, opt_d

    def training_step(self, batch, batch_idx):
        opt_ae, opt_g, opt_d = self.optimizers()
        x = batch
        b = x.size(0)

        opt_ae.zero_grad()
        z = self.enc(x)
        x_rec = self.dec(z)
        loss_recon = F.mse_loss(x_rec, x) * self.hparams.recon_w
        self.manual_backward(loss_recon)
        opt_ae.step()

        opt_d.zero_grad()
        z_real = torch.randn(b, self.hparams.z_dim, device=self.device)
        z_fake = self.enc(x).detach()
        d_real = self.disc(z_real)
        d_fake = self.disc(z_fake)
        loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                  + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
        self.manual_backward(loss_d)
        opt_d.step()

        opt_g.zero_grad()
        z_fake = self.enc(x)
        d_out = self.disc(z_fake)
        loss_g = F.binary_cross_entropy_with_logits(d_out, torch.ones_like(d_out)) * self.hparams.adv_w
        self.manual_backward(loss_g)
        opt_g.step()

        self._buf["recon"].append(loss_recon.item())
        self._buf["g_adv"].append(loss_g.item())
        self._buf["d_adv"].append(loss_d.item())
        self._last_batch = x.detach()

    def on_train_epoch_end(self):
        if self.run_dir is None:
            return
        avgs = {k: sum(v) / max(len(v), 1) for k, v in self._buf.items()}
        for v in self._buf.values():
            v.clear()

        with torch.no_grad():
            samples = self.dec(self.fixed_z.to(self.device))
        sample_std = float(samples.std(dim=0).mean().item())

        artifacts.log_metrics(
            self.run_dir,
            step=self.global_step, epoch=self.current_epoch, split="train",
            values={**avgs, "sample_std": sample_std},
        )
        artifacts.log_event(self.run_dir, "epoch_end", epoch=self.current_epoch, **avgs, sample_std=sample_std)
        self.log_dict({**avgs, "sample_std": sample_std}, on_epoch=True, prog_bar=False)

        save_image(samples * 0.5 + 0.5,
                   self.run_dir / "samples" / f"epoch_{self.current_epoch:03d}.png",
                   nrow=8)

        if self._last_batch is not None:
            x = self._last_batch[:8].to(self.device)
            with torch.no_grad():
                rec = self.dec(self.enc(x))
            grid = torch.cat([x, rec], dim=0)
            save_image(grid * 0.5 + 0.5,
                       self.run_dir / "samples" / f"recon_epoch_{self.current_epoch:03d}.png",
                       nrow=8)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.hparams.z_dim, device=self.device)
        return self.dec(z)
