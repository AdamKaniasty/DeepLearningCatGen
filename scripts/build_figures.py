from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from catgen import artifacts

FIG = artifacts.ROOT / "presentation" / "figures"
FAMILIES = ("dcgan", "aae", "vqvae")


def _split_name(m: dict) -> str:
    return (m.get("config") or {}).get("data", {}).get("split", "")


def is_mixed_run(m: dict) -> bool:
    return "mixed_train" in _split_name(m)


def load_done_runs() -> list[dict]:
    out = []
    for d in sorted(artifacts.RUNS.iterdir()):
        mp = d / "manifest.json"
        if not mp.exists() or d.name.startswith("_"):
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "done":
            continue
        fid = None
        fp = d / "eval" / "fid.json"
        if fp.exists():
            try:
                fid = json.loads(fp.read_text()).get("fid")
            except Exception:
                pass
        out.append({
            "dir": d,
            "model": m.get("model"),
            "fid": fid,
            "ds_hash": m.get("dataset_hash"),
            "mixed": is_mixed_run(m),
            "manifest": m,
        })
    return out


def cat_only_runs(runs: list[dict]) -> list[dict]:
    return [r for r in runs if not r["mixed"]]


def best_per_family(runs: list[dict], ds_hash: str | None = None) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in runs:
        if r["fid"] is None:
            continue
        if ds_hash is not None and r["ds_hash"] != ds_hash:
            continue
        cur = best.get(r["model"])
        if cur is None or r["fid"] < cur["fid"]:
            best[r["model"]] = r
    return best


def fig_fid_bar(runs: list[dict]) -> Path:
    best = best_per_family(cat_only_runs(runs))
    models = [m for m in FAMILIES if m in best]
    vals = [best[m]["fid"] for m in models]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(models, vals)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("FID (lower = better)")
    ax.set_title("Best FID per model family")
    fig.tight_layout()
    out = FIG / "fid_bar.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_compare_grid(runs: list[dict]) -> Path | None:
    best = best_per_family(cat_only_runs(runs))
    cols = []
    labels = []
    tf = transforms.ToTensor()
    for m in FAMILIES:
        if m not in best:
            continue
        last = sorted((best[m]["dir"] / "samples").glob("epoch_*.png"))
        if not last:
            continue
        cols.append(tf(Image.open(last[-1]).convert("RGB")))
        labels.append(m)
    if not cols:
        return None
    pad = 4
    h = max(c.shape[1] for c in cols)
    w = sum(c.shape[2] for c in cols) + pad * (len(cols) - 1)
    canvas = torch.ones(3, h, w)
    x = 0
    for c in cols:
        canvas[:, :c.shape[1], x:x + c.shape[2]] = c
        x += c.shape[2] + pad
    out = FIG / "compare_grid.png"
    save_image(canvas, out)
    return out


def _best_dcgan(runs: list[dict], *, mixed: bool) -> dict | None:
    pool = [r for r in runs if r["model"] == "dcgan" and r["mixed"] == mixed and r["fid"] is not None]
    if not pool:
        pool = [r for r in runs if r["model"] == "dcgan" and r["mixed"] == mixed]
    if not pool:
        return None
    with_fid = [r for r in pool if r["fid"] is not None]
    return min(with_fid or pool, key=lambda r: r["fid"] if r["fid"] is not None else 1e9)


def _last_sample(run: dict) -> Path | None:
    last = sorted((run["dir"] / "samples").glob("epoch_*.png"))
    return last[-1] if last else None


def fig_ext_compare(runs: list[dict]) -> Path | None:
    """Cat-only best DCGAN vs mixed-trained DCGAN (scope extension comparison)."""
    cat = _best_dcgan(runs, mixed=False)
    mix = _best_dcgan(runs, mixed=True)
    if cat is None or mix is None:
        return None
    cat_img, mix_img = _last_sample(cat), _last_sample(mix)
    if cat_img is None or mix_img is None:
        return None
    tf = transforms.ToTensor()
    cols = [tf(Image.open(cat_img).convert("RGB")), tf(Image.open(mix_img).convert("RGB"))]
    pad = 8
    h = max(c.shape[1] for c in cols)
    w = sum(c.shape[2] for c in cols) + pad
    canvas = torch.ones(3, h, w)
    x = 0
    for c in cols:
        canvas[:, :c.shape[1], x:x + c.shape[2]] = c
        x += c.shape[2] + pad
    out = FIG / "ext_compare.png"
    save_image(canvas, out)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_done_runs()
    if not runs:
        print("no done runs")
        return
    bar = fig_fid_bar(runs)
    print(f"wrote {bar}")
    grid = fig_compare_grid(runs)
    if grid:
        print(f"wrote {grid}")
    ext = fig_ext_compare(runs)
    if ext:
        print(f"wrote {ext}")


if __name__ == "__main__":
    main()
