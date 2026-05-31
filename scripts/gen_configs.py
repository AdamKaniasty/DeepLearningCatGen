from __future__ import annotations

import itertools
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "catgen" / "configs"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_N = 1500
REF_N = 500
TRAIN_SPLIT = f"train_{TRAIN_N}.txt"
MIXED_TRAIN_PER_CLASS = 400
MIXED_REF_PER_CLASS = 200


def emit(name: str, cfg: dict) -> None:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def clear_sweep_configs() -> None:
    for prefix in ("dcgan_", "aae_", "vqvae_"):
        for p in OUT.glob(f"{prefix}*.yaml"):
            if "smoke" not in p.name and "ext_" not in p.name:
                p.unlink()


def base_data(bs: int = 32, split: str | None = None) -> dict:
    return {
        "split": split or TRAIN_SPLIT,
        "image_size": 64,
        "batch_size": bs,
        "num_workers": 4,
        "augment": True,
    }


def dcgan_grid(epochs: int = 30) -> list[tuple[str, dict]]:
    out = []
    for z, lr in itertools.product([64, 128], [2e-4, 1e-4]):
        name = f"dcgan_z{z}_lr{lr:.0e}_bs32"
        cfg = {
            "model": "dcgan",
            "seed": 42,
            "max_epochs": epochs,
            "save_every": 5,
            "data": base_data(bs=32),
            "model_args": {"z_dim": z, "ch": 64, "lr": lr, "beta1": 0.5, "n_sample": 64},
        }
        out.append((name, cfg))
    return out


def aae_grid(epochs: int = 40) -> list[tuple[str, dict]]:
    out = []
    for z, lr in [(64, 1e-4), (128, 1e-4), (128, 1e-5)]:
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
    for K, lr in [(128, 2e-4), (512, 2e-4), (512, 1e-4)]:
        D = 64
        bs = 16 if K >= 512 else 32
        name = f"vqvae_K{K}_D{D}_lr{lr:.0e}"
        cfg = {
            "model": "vqvae",
            "seed": 42,
            "max_epochs": epochs,
            "data": base_data(bs=bs),
            "model_args": {"num_embeddings": K, "embedding_dim": D, "hidden": 128, "lr": lr},
            "early_stop": {"monitor": "recon", "patience": 8, "min_delta": 1.0e-4, "mode": "min"},
        }
        out.append((name, cfg))
    return out


def extension_dcgan(epochs: int = 30) -> tuple[str, dict]:
    """Cats+dogs extension: one DCGAN on balanced mixed split (scope: single selected model)."""
    n = 2 * MIXED_TRAIN_PER_CLASS
    name = "dcgan_ext_mixed_z128_lr2e-04_bs32"
    cfg = {
        "model": "dcgan",
        "seed": 42,
        "max_epochs": epochs,
        "save_every": 5,
        "data": base_data(bs=32, split=f"mixed_train_{n}.txt"),
        "model_args": {"z_dim": 128, "ch": 64, "lr": 2e-4, "beta1": 0.5, "n_sample": 64},
        "tags": ["extension", "mixed"],
    }
    return name, cfg


def main():
    clear_sweep_configs()
    configs = dcgan_grid() + aae_grid() + vqvae_grid()
    for name, cfg in configs:
        emit(name, cfg)
    ext_name, ext_cfg = extension_dcgan()
    emit(ext_name, ext_cfg)
    print(
        f"wrote {len(configs) + 1} configs to {OUT} "
        f"(train={TRAIN_SPLIT}, fid_ref=fid_ref_{REF_N}.txt, "
        f"mixed_train={2 * MIXED_TRAIN_PER_CLASS}, mixed_ref={2 * MIXED_REF_PER_CLASS})"
    )


if __name__ == "__main__":
    main()
