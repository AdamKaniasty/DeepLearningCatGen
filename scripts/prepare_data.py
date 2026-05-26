from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())


def hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode())
    return h.hexdigest()[:12]


def write_split(name: str, paths: list[Path]) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    rel = [str(p.relative_to(RAW)) for p in paths]
    (SPLITS / name).write_text("\n".join(rel) + "\n")


def make_fake(category: str, n: int) -> None:
    rng = random.Random(0)
    out = RAW / category
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        Image.new("RGB", (96, 96), c).save(out / f"{i:05d}.png")


def build_cat_splits(seed: int, train_n: int, ref_n: int) -> dict:
    cats = scan_images(RAW / "cats")
    if len(cats) < train_n + ref_n:
        raise SystemExit(f"need >= {train_n + ref_n} cat images, have {len(cats)}")
    rng = random.Random(seed)
    rng.shuffle(cats)
    train = sorted(cats[:train_n])
    ref = sorted(cats[train_n:train_n + ref_n])
    reserve = sorted(cats[train_n + ref_n:])
    write_split(f"train_{train_n}.txt", train)
    write_split(f"fid_ref_{ref_n}.txt", ref)
    write_split(f"reserve_{len(reserve)}.txt", reserve)
    return {
        "seed": seed,
        "source_total": len(cats),
        "train": {"file": f"train_{train_n}.txt", "n": train_n, "hash": hash_paths(train)},
        "fid_ref": {"file": f"fid_ref_{ref_n}.txt", "n": ref_n, "hash": hash_paths(ref)},
        "reserve": {"file": f"reserve_{len(reserve)}.txt", "n": len(reserve), "hash": hash_paths(reserve)},
    }


def build_mixed_splits(seed: int, per_class_train: int, per_class_ref: int) -> dict:
    cats = scan_images(RAW / "cats")
    dogs = scan_images(RAW / "dogs")
    need = per_class_train + per_class_ref
    if min(len(cats), len(dogs)) < need:
        raise SystemExit(f"need >= {need} per class, have cats={len(cats)} dogs={len(dogs)}")
    rng = random.Random(seed)
    rng.shuffle(cats)
    rng.shuffle(dogs)
    train = sorted(cats[:per_class_train] + dogs[:per_class_train])
    ref = sorted(cats[per_class_train:need] + dogs[per_class_train:need])
    write_split(f"mixed_train_{2 * per_class_train}.txt", train)
    write_split(f"mixed_ref_{2 * per_class_ref}.txt", ref)
    return {
        "seed": seed,
        "mixed_train": {"file": f"mixed_train_{2 * per_class_train}.txt", "n": len(train), "hash": hash_paths(train)},
        "mixed_ref": {"file": f"mixed_ref_{2 * per_class_ref}.txt", "n": len(ref), "hash": hash_paths(ref)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", type=int, default=0, help="generate N fake cat images for smoke")
    ap.add_argument("--fake-dogs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-n", type=int, default=3000)
    ap.add_argument("--ref-n", type=int, default=1000)
    ap.add_argument("--mixed-train", type=int, default=1000, help="per-class")
    ap.add_argument("--mixed-ref", type=int, default=500, help="per-class")
    ap.add_argument("--no-mixed", action="store_true")
    args = ap.parse_args()

    if args.fake:
        make_fake("cats", args.fake)
    if args.fake_dogs:
        make_fake("dogs", args.fake_dogs)

    info = {"cat": build_cat_splits(args.seed, args.train_n, args.ref_n)}
    if not args.no_mixed and (RAW / "dogs").exists():
        info["mixed"] = build_mixed_splits(args.seed, args.mixed_train, args.mixed_ref)

    (SPLITS / "manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
