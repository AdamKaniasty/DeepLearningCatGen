from __future__ import annotations

import argparse
import json

import torch
from torchvision.utils import save_image

from catgen import artifacts
from catgen.data import CatDataModule
from catgen.train import MODELS, _register, pick_accelerator


def device_for(acc: str) -> torch.device:
    if acc == "cuda":
        return torch.device("cuda")
    if acc == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    _register()
    d = artifacts.RUNS / args.run_id
    manifest = json.loads((d / "manifest.json").read_text())
    cfg = manifest["config"]
    model_name = manifest["model"]
    cls = MODELS[model_name]

    lm = cls.load_from_checkpoint(d / "checkpoints" / "last.ckpt")
    dev = device_for(pick_accelerator(args.device))
    lm.to(dev).eval()

    out = d / "eval" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()

    written = 0
    if model_name == "vqvae":
        dm = CatDataModule(**cfg["data"])
        dm.setup()
        freq = lm.empirical_code_freq(dm.train_dataloader(), dev)
        grid = cfg["data"]["image_size"] // 4
        while written < args.n:
            n = min(args.batch, args.n - written)
            imgs = lm.sample_from_freq(n, freq, grid, dev)
            for j in range(n):
                save_image(imgs[j] * 0.5 + 0.5, out / f"{written:05d}.png")
                written += 1
    else:
        while written < args.n:
            n = min(args.batch, args.n - written)
            imgs = lm.sample(n)
            for j in range(n):
                save_image(imgs[j] * 0.5 + 0.5, out / f"{written:05d}.png")
                written += 1

    artifacts.log_event(d, "samples_generated", n=written, dir="eval/samples")
    print(f"wrote {written} samples to {out}")


if __name__ == "__main__":
    main()
