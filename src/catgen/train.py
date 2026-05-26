from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from catgen import artifacts
from catgen.data import CatDataModule, SPLITS

MODELS = {}


def _register():
    from catgen.lightning_modules.gan_lm import GAN
    from catgen.lightning_modules.aae_lm import AAE
    from catgen.lightning_modules.vqvae_lm import VQ
    MODELS["dcgan"] = GAN
    MODELS["aae"] = AAE
    MODELS["vqvae"] = VQ


def pick_accelerator(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def dataset_hash(split_name: str) -> str | None:
    p = SPLITS / split_name
    if not p.exists():
        return None
    import hashlib
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None, help="for smoke runs")
    ap.add_argument("--force", action="store_true", help="rerun even if manifest status=done")
    ap.add_argument("--set", action="append", default=[], help="override config: e.g. data.split=train_30.txt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        d = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = v

    _register()
    model_name = cfg["model"]
    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    rid = artifacts.run_id(model_name, cfg, seed)
    if artifacts.is_done(rid) and not args.force:
        print(f"[skip] {rid} already done")
        return

    d = artifacts.run_dir(rid)
    artifacts.write_manifest(d, model=model_name, config=cfg, seed=seed, dataset_hash=dataset_hash(cfg["data"]["split"]))

    dm = CatDataModule(**cfg["data"])
    lm = MODELS[model_name](**cfg.get("model_args", {}))
    lm.run_dir = d

    n_params = {name: sum(p.numel() for p in mod.parameters()) for name, mod in lm.named_children()}
    n_params["total"] = sum(n_params.values())

    acc = pick_accelerator(args.device)
    callbacks = []
    es_cfg = cfg.get("early_stop")
    if es_cfg:
        callbacks.append(EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            mode=es_cfg.get("mode", "min"),
            check_on_train_epoch_end=True,
            verbose=True,
        ))

    save_every = int(cfg.get("save_every", 0))
    enable_ckpt = save_every > 0
    if enable_ckpt:
        callbacks.append(ModelCheckpoint(
            dirpath=str(d / "checkpoints"),
            filename="epoch_{epoch:03d}",
            save_top_k=-1,
            every_n_epochs=save_every,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ))

    trainer = L.Trainer(
        accelerator=acc,
        devices=1,
        max_epochs=cfg["max_epochs"],
        limit_train_batches=args.limit_batches or 1.0,
        enable_checkpointing=enable_ckpt,
        enable_progress_bar=True,
        logger=False,
        deterministic=False,
        callbacks=callbacks,
    )

    try:
        trainer.fit(lm, datamodule=dm)
    except Exception as e:
        artifacts.log_event(d, "training_failed", error=repr(e))
        artifacts.mark_done(d, status="failed", error=repr(e))
        raise

    for cb in callbacks:
        if isinstance(cb, EarlyStopping) and cb.stopped_epoch:
            artifacts.log_event(d, "early_stopped", epoch=int(cb.stopped_epoch), monitor=cb.monitor)

    ckpt = d / "checkpoints" / "last.ckpt"
    trainer.save_checkpoint(ckpt)

    last_grids = sorted((d / "samples").glob("epoch_*.png"))
    summary = [
        f"# Run {rid}",
        "",
        f"- model: **{model_name}**",
        f"- seed: {seed}",
        f"- epochs: {cfg['max_epochs']}",
        f"- accelerator: {acc}",
        f"- dataset_hash: {dataset_hash(cfg['data']['split'])}",
        f"- checkpoint: `{ckpt.relative_to(d)}`",
        f"- params: {n_params}",
        "",
        "## Config",
        "```yaml",
        yaml.safe_dump(cfg, sort_keys=False).strip(),
        "```",
    ]
    if last_grids:
        summary += ["", "## Last sample grid", f"![samples](samples/{last_grids[-1].name})"]
    artifacts.write_summary(d, "\n".join(summary) + "\n")
    artifacts.mark_done(d, n_params=n_params)
    print(f"[done] {rid}")


if __name__ == "__main__":
    main()
