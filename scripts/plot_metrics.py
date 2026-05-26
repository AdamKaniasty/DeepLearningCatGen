from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catgen import artifacts


def read_metrics(p: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                series[row["name"]].append((int(row["epoch"]), float(row["value"])))
            except Exception:
                continue
    return series


def plot_run(run_dir: Path) -> Path | None:
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None
    series = read_metrics(metrics)
    if not series:
        return None
    n = len(series)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, (name, vals) in enumerate(sorted(series.items())):
        ax = axes[i // cols][i % cols]
        xs, ys = zip(*vals)
        ax.plot(xs, ys, marker="o", ms=3)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    out = run_dir / "eval" / "curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="single run; default = all done runs")
    args = ap.parse_args()
    targets = [artifacts.RUNS / args.run_id] if args.run_id else [
        d for d in artifacts.RUNS.iterdir() if (d / "manifest.json").exists() and not d.name.startswith("_")
    ]
    n_ok = 0
    for d in targets:
        out = plot_run(d)
        if out:
            print(f"wrote {out}")
            n_ok += 1
    print(f"done: {n_ok}/{len(targets)}")


if __name__ == "__main__":
    main()
