#!/usr/bin/env python3
"""Check that presentation/runs.yaml assets exist."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "presentation" / "runs.yaml"


def resolve(path: str, runs: dict) -> Path:
    flat = {}
    for section, mapping in runs.items():
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                flat[f"{section}.{k}"] = v
    for key, val in flat.items():
        if val is not None:
            path = path.replace(f"{{{key}}}", str(val))
    return ROOT / path


STAGED = [
    "presentation/assets/dataset/dataset_preview.png",
    "presentation/assets/part64/fid_bar.png",
    "presentation/assets/part64/compare_grid.png",
    "presentation/assets/part64/dcgan_samples.png",
    "presentation/assets/part64/ext_compare.png",
    "presentation/assets/part64/interp_dcgan.png",
    "presentation/assets/part64/interp_aae.png",
    "presentation/assets/part128/fid_bar.png",
    "presentation/assets/part128/compare_grid.png",
    "presentation/assets/part128/hero_dcgan.png",
    "presentation/assets/part128/interp_dcgan.png",
    "presentation/assets/part128/interp_aae.png",
    "presentation/assets/part128/ext_compare.png",
    "presentation/assets/partc/fid_bar.png",
    "presentation/assets/partc/gd_compare.png",
    "presentation/assets/partc/compare_grid.png",
    "presentation/assets/partc/interp_dcgan.png",
]


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    run_map = cfg.get("runs", {})
    missing = [p for p in STAGED if not (ROOT / p).is_file()]
    for section, items in cfg.get("assets", {}).items():
        if not isinstance(items, dict):
            continue
        for name, spec in items.items():
            if name.endswith("_n") or spec is None:
                continue
            if isinstance(spec, str) and "{" in spec:
                p = resolve(spec, run_map)
            else:
                p = ROOT / spec
            if name.endswith("_dir"):
                if not p.is_dir() or not any(p.glob("*.png")):
                    missing.append(f"{section}.{name}: {p}")
                continue
            if not p.is_file():
                missing.append(f"{section}.{name}: {p}")
    if missing:
        print("MISSING assets:")
        for m in missing:
            print(f"  {m}")
        return 1
    print(f"OK all assets from {CFG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
