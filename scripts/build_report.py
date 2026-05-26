from __future__ import annotations

import json
from pathlib import Path

from catgen import artifacts

REPORTS = artifacts.ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    sections = ["# Report Bundle", "", "Auto-generated summary of all training + eval artifacts.", ""]

    lb_md = REPORTS / "leaderboard.md"
    if lb_md.exists():
        sections += ["## Leaderboard", "", lb_md.read_text(), ""]

    by_model: dict[str, list[dict]] = {}
    for d in sorted(artifacts.RUNS.iterdir()):
        mp = d / "manifest.json"
        if not mp.exists() or d.name.startswith("_"):
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "done":
            continue
        by_model.setdefault(m.get("model", "unknown"), []).append({
            "rid": d.name,
            "manifest": m,
            "dir": d,
        })

    for model, runs in by_model.items():
        sections += [f"## {model.upper()} ({len(runs)} runs)", ""]
        for r in runs:
            d = r["dir"]
            fid = "-"
            fp = d / "eval" / "fid.json"
            if fp.exists():
                try:
                    fid = f"{json.loads(fp.read_text())['fid']:.3f}"
                except Exception:
                    pass
            ds_hash = r["manifest"].get("dataset_hash")
            sections += [
                f"### `{r['rid']}`",
                "",
                f"- model: {model}",
                f"- dataset_hash: `{ds_hash}`",
                f"- FID: **{fid}**",
                f"- summary: [{r['rid']}/summary.md](../runs/{r['rid']}/summary.md)",
            ]
            last_grid = sorted((d / "samples").glob("epoch_*.png"))
            if last_grid:
                sections += [f"- last sample grid: ![](../runs/{r['rid']}/samples/{last_grid[-1].name})"]
            interp = d / "eval" / "interpolation.png"
            if interp.exists():
                sections += [f"- interpolation: ![](../runs/{r['rid']}/eval/interpolation.png)"]
            sections += [""]

    (REPORTS / "report_bundle.md").write_text("\n".join(sections) + "\n")
    print(f"wrote {REPORTS / 'report_bundle.md'}")


if __name__ == "__main__":
    main()
