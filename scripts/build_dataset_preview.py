#!/usr/bin/env python3
"""Build dataset preview for slides (real images, not smoke placeholders)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
OUT_DIR = ROOT / "presentation" / "figures"
MIN_BYTES = 2048


def _paths_ok(paths: list[Path]) -> bool:
    return bool(paths) and all(p.is_file() and p.stat().st_size >= MIN_BYTES for p in paths)


def grid_from_split(split: str, n: int, size: int, seed: int) -> torch.Tensor | None:
    split_path = SPLITS / split
    if not split_path.is_file():
        return None
    rels = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    paths = [RAW / r for r in rels]
    paths = [p for p in paths if p.is_file() and p.stat().st_size >= MIN_BYTES]
    if len(paths) < n:
        return None
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:n]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def grid_from_urls(n_cats: int, size: int, seed: int) -> torch.Tensor:
    """Fallback when local data/raw are smoke placeholders (no real Kaggle sync)."""
    import subprocess

    cache = OUT_DIR / "_preview_cache"
    cache.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    rng = random.Random(seed)
    tiles = []
    for k in range(n_cats):
        path = cache / f"{k:02d}.jpg"
        if not path.is_file() or path.stat().st_size < MIN_BYTES:
            url = f"https://cataas.com/cat?{rng.randint(0, 10**9)}"
            subprocess.run(["curl", "-sfL", url, "-o", str(path)], check=True)
        tiles.append(tf(Image.open(path).convert("RGB")))
    return torch.stack(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-split", default="train_3000.txt")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_cats, n_dogs = 16, 8

    cats = grid_from_split(args.cat_split, n_cats, args.size, args.seed)
    if cats is None:
        cats = grid_from_split("train_30.txt", n_cats, args.size, args.seed)
    if cats is None:
        print("local raw images are smoke placeholders; fetching representative cat photos")
        cats = grid_from_urls(n_cats, args.size, args.seed)
        save_image(cats, OUT_DIR / "dataset_preview_cats.png", nrow=8)
        save_image(cats, OUT_DIR / "dataset_preview.png", nrow=8)
        print(f"wrote {OUT_DIR / 'dataset_preview.png'} (fallback URLs; replace after rsync)")
        return

    save_image(cats, OUT_DIR / "dataset_preview_cats.png", nrow=8)
    save_image(cats, OUT_DIR / "dataset_preview.png", nrow=8)
    print(f"wrote {OUT_DIR / 'dataset_preview.png'} ({len(cats)} cats from split)")


if __name__ == "__main__":
    main()
