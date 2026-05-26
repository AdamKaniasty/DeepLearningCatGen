from __future__ import annotations

import itertools
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "catgen" / "configs"
OUT.mkdir(parents=True, exist_ok=True)


def emit(name: str, cfg: dict) -> None:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def base_data(split: str = "train_3000.txt", bs: int = 32) -> dict:
    return {"split": split, "image_size": 64, "batch_size": bs, "num_workers": 4, "augment": True}


def dcgan_grid(epochs: int = 30) -> list[tuple[str, dict]]:
    out = []
    for z, lr, bs in itertools.product([64, 128], [2e-4, 1e-4], [32, 64]):
        name = f"dcgan_z{z}_lr{lr:.0e}_bs{bs}"
        cfg = {
            "model": "dcgan",
            "seed": 42,
            "max_epochs": epochs,
            "save_every": 5,
            "data": base_data(bs=bs),
            "model_args": {"z_dim": z, "ch": 64, "lr": lr, "beta1": 0.5, "n_sample": 64},
        }
        out.append((name, cfg))
    return out


def aae_grid(epochs: int = 40) -> list[tuple[str, dict]]:
    out = []
    for z, lr in itertools.product([64, 128], [1e-4, 1e-5]):
        name = f"aae_z{z}_lr{lr:.0e}"
        cfg = {
            "model": "aae",
            "seed": 42,
            "max_epochs": epochs,
            "data": base_data(bs=32),
            "model_args": {"z_dim": z, "ch": 64, "lr": lr, "n_sample": 64},
            "early_stop": {"monitor": "recon", "patience": 8, "min_delta": 1.0e-4, "mode": "min"},
        }
        out.append((name, cfg))
    return out


def vqvae_grid(epochs: int = 40) -> list[tuple[str, dict]]:
    out = []
    for K, D, lr in itertools.product([128, 512], [64, 128], [1e-4, 2e-4]):
        name = f"vqvae_K{K}_D{D}_lr{lr:.0e}"
        cfg = {
            "model": "vqvae",
            "seed": 42,
            "max_epochs": epochs,
            "data": base_data(bs=32),
            "model_args": {"num_embeddings": K, "embedding_dim": D, "hidden": 128, "lr": lr},
            "early_stop": {"monitor": "recon", "patience": 8, "min_delta": 1.0e-4, "mode": "min"},
        }
        out.append((name, cfg))
    return out


def main():
    configs = dcgan_grid() + aae_grid() + vqvae_grid()
    for name, cfg in configs:
        emit(name, cfg)
    print(f"wrote {len(configs)} configs to {OUT}")


if __name__ == "__main__":
    main()
