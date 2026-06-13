"""Part C: DCGAN @128 with rebalanced G vs D (same data as phase128 Part B)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "catgen" / "configs"

DATA_128 = {
    "image_size": 128,
    "num_workers": 4,
    "augment": True,
    "augment_crop": True,
    "split": "train_3000.txt",
    "batch_size": 32,
}


def main() -> None:
    # Phase B baseline: lr=1e-4, lr_d=2e-4, label_smooth=0.1  -> dcgan_e9574605_42
    cfg = {
        "model": "dcgan",
        "seed": 42,
        "max_epochs": 80,
        "save_every": 5,
        "tags": ["partc"],
        "data": dict(DATA_128),
        "model_args": {
            "z_dim": 128,
            "ch": 64,
            "image_size": 128,
            "lr": 2e-4,
            "lr_d": 1e-4,
            "beta1": 0.5,
            "label_smooth": 0.05,
            "generator": "upsample_conv",
            "n_sample": 64,
        },
    }
    path = OUT / "dcgan_128_partc_gd_rebalance.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {path}")
    print("  vs Phase B: lr 1e-4->2e-4, lr_d 2e-4->1e-4, label_smooth 0.1->0.05")


if __name__ == "__main__":
    main()
