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
        out.append({"dir": d, "model": m.get("model"), "fid": fid, "ds_hash": m.get("dataset_hash")})
    return out


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
    best = best_per_family(runs)
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
    best = best_per_family(runs)
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


def fig_ext_compare(runs: list[dict]) -> Path | None:
    by_hash: dict[str, list[dict]] = {}
    for r in runs:
        by_hash.setdefault(r["ds_hash"] or "?", []).append(r)
    if len(by_hash) < 2:
        return None

    hashes = list(by_hash.keys())
    cells: dict[tuple[str, str], Path] = {}
    tf = transforms.ToTensor()
    for h in hashes:
        best = best_per_family(by_hash[h])
        for m, r in best.items():
            last = sorted((r["dir"] / "samples").glob("epoch_*.png"))
            if last:
                cells[(h, m)] = last[-1]
    if not cells:
        return None

    rows = []
    for h in hashes:
        row_imgs = [tf(Image.open(cells[(h, m)]).convert("RGB")) for m in FAMILIES if (h, m) in cells]
        if not row_imgs:
            continue
        w = sum(im.shape[2] for im in row_imgs) + 4 * (len(row_imgs) - 1)
        h_px = max(im.shape[1] for im in row_imgs)
        row = torch.ones(3, h_px, w)
        x = 0
        for im in row_imgs:
            row[:, :im.shape[1], x:x + im.shape[2]] = im
            x += im.shape[2] + 4
        rows.append(row)

    pad = 8
    H = sum(r.shape[1] for r in rows) + pad * (len(rows) - 1)
    W = max(r.shape[2] for r in rows)
    canvas = torch.ones(3, H, W)
    y = 0
    for r in rows:
        canvas[:, y:y + r.shape[1], :r.shape[2]] = r
        y += r.shape[1] + pad
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
