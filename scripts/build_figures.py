from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from catgen import artifacts

PHASE = os.environ.get("PRESENTATION_PHASE", "sweep")  # sweep | refine | phase128
FIG = artifacts.ROOT / "presentation" / "figures"
if PHASE == "phase128":
    FIG = FIG / "phase128"
FAMILIES = ("dcgan", "aae", "vqvae")


def _split_name(m: dict) -> str:
    return (m.get("config") or {}).get("data", {}).get("split", "")


def is_mixed_run(m: dict) -> bool:
    return "mixed_train" in _split_name(m)


def is_refine_run(m: dict) -> bool:
    return "refine" in (m.get("config") or {}).get("tags", [])


def is_phase128_run(m: dict) -> bool:
    tags = (m.get("config") or {}).get("tags", [])
    if "phase128" in tags or "dcgan128" in tags:
        return True
    data = (m.get("config") or {}).get("data", {})
    split = data.get("split", "")
    return data.get("image_size") == 128 and "train_3000" in split


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
            "refine": is_refine_run(m),
            "phase128": is_phase128_run(m),
            "manifest": m,
        })
    return out


def phase_runs(runs: list[dict]) -> list[dict]:
    if PHASE == "phase128":
        return [r for r in runs if r["phase128"]]
    if PHASE == "refine":
        return [r for r in runs if r["refine"]]
    return [r for r in runs if not r["refine"] and not r["phase128"]]


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
    title = "Best FID per model family"
    if PHASE == "refine":
        title += " (refinement phase)"
    elif PHASE == "phase128":
        title += " (128×128, 3000 cats)"
    ax.set_title(title)
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


def _best_model_run(runs: list[dict], model: str, *, mixed: bool) -> dict | None:
    pool = [r for r in runs if r["model"] == model and r["mixed"] == mixed]
    if not pool:
        return None
    with_fid = [r for r in pool if r["fid"] is not None]
    return min(with_fid or pool, key=lambda r: r["fid"] if r["fid"] is not None else 1e9)


def _overall_best_cats_model(runs: list[dict]) -> str | None:
    best_fid, best_model = 1e9, None
    for r in cat_only_runs(runs):
        if r["fid"] is None:
            continue
        if r["fid"] < best_fid:
            best_fid, best_model = r["fid"], r["model"]
    return best_model


def _sample_panel(run: dict, n: int = 16, nrow: int = 4) -> torch.Tensor | None:
    """4x4 (or similar) grid from eval/samples for readable slides."""
    eval_dir = run["dir"] / "eval" / "samples"
    paths = sorted(eval_dir.glob("*.png"))[:n] if eval_dir.is_dir() else []
    if len(paths) < n:
        last = sorted((run["dir"] / "samples").glob("epoch_*.png"))
        if not last:
            return None
        img = Image.open(last[-1]).convert("RGB")
        w, h = img.size
        side = int(n**0.5)
        cell = w // side
        tiles = []
        for row in range(side):
            for col in range(side):
                box = (col * cell, row * cell, (col + 1) * cell, (row + 1) * cell)
                tiles.append(img.crop(box))
        paths = None
        imgs = [transforms.ToTensor()(t) for t in tiles[:n]]
    else:
        imgs = [transforms.ToTensor()(Image.open(p).convert("RGB")) for p in paths]
    if not imgs:
        return None
    from torchvision.utils import make_grid
    return make_grid(imgs, nrow=nrow, padding=2, normalize=False)


def fig_ext_compare(runs: list[dict]) -> Path | None:
    """Cats-only vs mixed for best model (DCGAN in sweep/refine; overall best in phase128)."""
    if PHASE == "phase128":
        model = _overall_best_cats_model(runs) or "dcgan"
        cat = _best_model_run(runs, model, mixed=False)
        mix = _best_model_run(runs, model, mixed=True)
    else:
        model = "dcgan"
        cat = _best_dcgan(runs, mixed=False)
        mix = _best_dcgan(runs, mixed=True)
    if cat is None or mix is None:
        return None
    cat_panel = _sample_panel(cat)
    mix_panel = _sample_panel(mix)
    if cat_panel is None or mix_panel is None:
        return None

    def _fid_label(run: dict) -> str:
        fp = run["dir"] / "eval" / "fid.json"
        if fp.exists():
            try:
                v = json.loads(fp.read_text()).get("fid")
                if v is not None:
                    return f"FID={v:.1f}"
            except Exception:
                pass
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    tf = transforms.ToPILImage()
    titles = [
        f"Cats only ({model.upper()})\n{_fid_label(cat)}",
        f"Cats + dogs ({model.upper()})\n{_fid_label(mix)}",
    ]
    for ax, panel, title in zip(axes, [cat_panel, mix_panel], titles):
        ax.imshow(tf(panel))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    if PHASE == "phase128":
        phase_note = " — 128×128, best of three families"
    elif PHASE == "refine":
        phase_note = " (refined)"
    else:
        phase_note = ""
    fig.suptitle(
        f"Extension: same architecture, different training data{phase_note}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = FIG / "ext_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_interp_copy(runs: list[dict], model: str, out_name: str) -> Path | None:
    best = best_per_family(cat_only_runs(runs))
    if model not in best:
        return None
    src = best[model]["dir"] / "eval" / "interpolation.png"
    if not src.exists():
        return None
    out = FIG / out_name
    out.write_bytes(src.read_bytes())
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    all_runs = load_done_runs()
    runs = phase_runs(all_runs)
    if not runs:
        print(f"no done runs for phase={PHASE}")
        return
    print(f"build_figures phase={PHASE} ({len(runs)} runs)")
    bar = fig_fid_bar(runs)
    print(f"wrote {bar}")
    grid = fig_compare_grid(runs)
    if grid:
        print(f"wrote {grid}")
    ext = fig_ext_compare(runs)
    if ext:
        print(f"wrote {ext}")
    for model, name in (("dcgan", "interp_dcgan.png"), ("aae", "interp_aae.png")):
        p = fig_interp_copy(runs, model, name)
        if p:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
