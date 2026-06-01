"""Configs for 128x128 phase (3000 cats + mixed extension variants)."""
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
}
EPOCHS = 80


def emit(name: str, cfg: dict) -> None:
    path = OUT / (name if name.endswith(".yaml") else f"{name}.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    emit(
        "aae_128_t3000_z128_bs32",
        {
            "model": "aae",
            "seed": 42,
            "max_epochs": EPOCHS,
            "tags": ["phase128"],
            "data": {**DATA_128, "split": "train_3000.txt", "batch_size": 32},
            "model_args": {"z_dim": 128, "ch": 64, "image_size": 128, "lr": 1e-4, "n_sample": 64},
            "early_stop": {"monitor": "recon", "patience": 10, "min_delta": 1e-4, "mode": "min"},
        },
    )
    emit(
        "vqvae_128_t3000_K128_bs16",
        {
            "model": "vqvae",
            "seed": 42,
            "max_epochs": EPOCHS,
            "tags": ["phase128"],
            "data": {**DATA_128, "split": "train_3000.txt", "batch_size": 16},
            "model_args": {
                "num_embeddings": 128,
                "embedding_dim": 64,
                "hidden": 128,
                "image_size": 128,
                "lr": 2e-4,
            },
            "early_stop": {"monitor": "recon", "patience": 10, "min_delta": 1e-4, "mode": "min"},
        },
    )
    dcgan_ext = {
        "model": "dcgan",
        "seed": 42,
        "max_epochs": EPOCHS,
        "save_every": 5,
        "tags": ["phase128", "extension", "mixed"],
        "data": {**DATA_128, "split": "mixed_train_800.txt", "batch_size": 32},
        "model_args": {
            "z_dim": 128,
            "ch": 64,
            "image_size": 128,
            "lr": 1e-4,
            "lr_d": 2e-4,
            "beta1": 0.5,
            "label_smooth": 0.1,
            "generator": "upsample_conv",
            "n_sample": 64,
        },
    }
    aae_ext = {
        "model": "aae",
        "seed": 42,
        "max_epochs": EPOCHS,
        "tags": ["phase128", "extension", "mixed"],
        "data": {**DATA_128, "split": "mixed_train_800.txt", "batch_size": 32},
        "model_args": {"z_dim": 128, "ch": 64, "image_size": 128, "lr": 1e-4, "n_sample": 64},
        "early_stop": {"monitor": "recon", "patience": 10, "min_delta": 1e-4, "mode": "min"},
    }
    vqvae_ext = {
        "model": "vqvae",
        "seed": 42,
        "max_epochs": EPOCHS,
        "tags": ["phase128", "extension", "mixed"],
        "data": {**DATA_128, "split": "mixed_train_800.txt", "batch_size": 16},
        "model_args": {
            "num_embeddings": 128,
            "embedding_dim": 64,
            "hidden": 128,
            "image_size": 128,
            "lr": 2e-4,
        },
        "early_stop": {"monitor": "recon", "patience": 10, "min_delta": 1e-4, "mode": "min"},
    }
    emit("dcgan_ext_128_mixed_bs32", dcgan_ext)
    emit("aae_ext_128_mixed_bs32", aae_ext)
    emit("vqvae_ext_128_mixed_bs16", vqvae_ext)
    print(f"wrote phase128 configs to {OUT}")


if __name__ == "__main__":
    main()
