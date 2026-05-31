"""Emit refinement-phase configs (Strategy B) — tags: [refine]."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "catgen" / "configs"
TRAIN_SPLIT = "train_1500.txt"
REFINE_EPOCHS = 80


def emit(name: str, cfg: dict) -> None:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    emit(
        "dcgan_refine_z128_bs64",
        {
            "model": "dcgan",
            "seed": 42,
            "max_epochs": REFINE_EPOCHS,
            "save_every": 5,
            "tags": ["refine"],
            "data": {
                "split": TRAIN_SPLIT,
                "image_size": 64,
                "batch_size": 64,
                "num_workers": 4,
                "augment": True,
            },
            "model_args": {
                "z_dim": 128,
                "ch": 64,
                "lr": 1e-4,
                "lr_d": 2e-4,
                "beta1": 0.5,
                "label_smooth": 0.1,
                "generator": "upsample_conv",
                "n_sample": 64,
            },
        },
    )
    emit(
        "aae_refine_z128_bs64",
        {
            "model": "aae",
            "seed": 42,
            "max_epochs": REFINE_EPOCHS,
            "tags": ["refine"],
            "data": {
                "split": TRAIN_SPLIT,
                "image_size": 64,
                "batch_size": 64,
                "num_workers": 4,
                "augment": True,
            },
            "model_args": {"z_dim": 128, "ch": 64, "lr": 1e-4, "n_sample": 64},
            "early_stop": {
                "monitor": "recon",
                "patience": 10,
                "min_delta": 1.0e-4,
                "mode": "min",
            },
        },
    )
    emit(
        "vqvae_refine_K128_bs32",
        {
            "model": "vqvae",
            "seed": 42,
            "max_epochs": REFINE_EPOCHS,
            "tags": ["refine"],
            "data": {
                "split": TRAIN_SPLIT,
                "image_size": 64,
                "batch_size": 32,
                "num_workers": 4,
                "augment": True,
            },
            "model_args": {
                "num_embeddings": 128,
                "embedding_dim": 64,
                "hidden": 128,
                "lr": 2e-4,
            },
            "early_stop": {
                "monitor": "recon",
                "patience": 10,
                "min_delta": 1.0e-4,
                "mode": "min",
            },
        },
    )
    emit(
        "dcgan_ext_refine_mixed_bs64",
        {
            "model": "dcgan",
            "seed": 42,
            "max_epochs": REFINE_EPOCHS,
            "save_every": 5,
            "tags": ["refine", "extension", "mixed"],
            "data": {
                "split": "mixed_train_800.txt",
                "image_size": 64,
                "batch_size": 64,
                "num_workers": 4,
                "augment": True,
            },
            "model_args": {
                "z_dim": 128,
                "ch": 64,
                "lr": 2e-4,
                "lr_d": 2e-4,
                "beta1": 0.5,
                "label_smooth": 0.1,
                "generator": "upsample_conv",
                "n_sample": 64,
            },
        },
    )
    print(f"wrote 4 refine configs to {OUT}")


if __name__ == "__main__":
    main()
