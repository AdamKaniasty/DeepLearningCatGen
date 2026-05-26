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


def grid_from_split(split: str, n: int, size: int, seed: int) -> torch.Tensor:
    paths = [RAW / line.strip() for line in (SPLITS / split).read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:n]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-split", default="train_3000.txt")
    ap.add_argument("--dog-split", default=None)
    ap.add_argument("--n-cats", type=int, default=16)
    ap.add_argument("--n-dogs", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cats = grid_from_split(args.cat_split, args.n_cats, args.size, args.seed)
    save_image(cats, OUT_DIR / "dataset_preview_cats.png", nrow=8)
    print(f"wrote {OUT_DIR / 'dataset_preview_cats.png'} ({len(cats)} cats)")

    if args.dog_split and (SPLITS / args.dog_split).exists():
        dogs = grid_from_split(args.dog_split, args.n_dogs, args.size, args.seed)
        save_image(dogs, OUT_DIR / "dataset_preview_dogs.png", nrow=8)
        print(f"wrote {OUT_DIR / 'dataset_preview_dogs.png'} ({len(dogs)} dogs)")
        combined = torch.cat([cats[:args.n_cats], dogs[:args.n_dogs]], dim=0)
        save_image(combined, OUT_DIR / "dataset_preview.png", nrow=8)
        print(f"wrote {OUT_DIR / 'dataset_preview.png'}")
    else:
        save_image(cats, OUT_DIR / "dataset_preview.png", nrow=8)


if __name__ == "__main__":
    main()
