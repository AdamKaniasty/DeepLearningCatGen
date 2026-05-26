from __future__ import annotations

import argparse
import json

import torch
from torchvision.utils import save_image

from catgen import artifacts
from catgen.sample import device_for
from catgen.train import MODELS, _register, pick_accelerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    _register()
    d = artifacts.RUNS / args.run_id
    manifest = json.loads((d / "manifest.json").read_text())
    model_name = manifest["model"]
    if model_name == "vqvae":
        artifacts.log_event(d, "interpolation_skipped", reason="discrete-latents")
        print("vqvae: skipping (discrete latents)")
        return

    cls = MODELS[model_name]
    lm = cls.load_from_checkpoint(d / "checkpoints" / "last.ckpt")
    dev = device_for(pick_accelerator(args.device))
    lm.to(dev).eval()

    z_dim = lm.hparams.z_dim
    z_a = torch.randn(1, z_dim, device=dev)
    z_b = torch.randn(1, z_dim, device=dev)
    alphas = torch.linspace(0, 1, args.steps, device=dev).view(-1, 1)
    z = (1 - alphas) * z_a + alphas * z_b

    with torch.no_grad():
        if model_name == "dcgan":
            imgs = lm.G(z)
        else:
            imgs = lm.dec(z)
    out = d / "eval" / "interpolation.png"
    save_image(imgs * 0.5 + 0.5, out, nrow=args.steps)
    artifacts.log_event(d, "interpolation_saved", steps=args.steps)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
