from __future__ import annotations

import csv
import json

from catgen import artifacts

REPORTS = artifacts.ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for d in sorted(artifacts.RUNS.iterdir()):
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "done" or d.name.startswith("_"):
            continue
        fid_p = d / "eval" / "fid.json"
        fid = None
        if fid_p.exists():
            try:
                fid = json.loads(fid_p.read_text()).get("fid")
            except Exception:
                pass
        sharp = div = nn_min = None
        qp = d / "eval" / "quality.json"
        if qp.exists():
            try:
                q = json.loads(qp.read_text())
                sharp = q["sharpness_laplacian_var"]["mean"]
                div = q["diversity_pairwise_inception"]["mean"]
                nn_min = q["nn_to_train_inception"]["min"]
            except Exception:
                pass
        rows.append({
            "run_id": d.name,
            "model": m.get("model"),
            "fid": fid,
            "sharpness": sharp,
            "diversity": div,
            "nn_min": nn_min,
            "epochs": m.get("config", {}).get("max_epochs"),
            "seed": m.get("seed"),
            "dataset_hash": m.get("dataset_hash"),
        })

    fields = ["run_id", "model", "fid", "sharpness", "diversity", "nn_min", "epochs", "seed", "dataset_hash"]
    with (REPORTS / "leaderboard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rows_sorted = sorted(rows, key=lambda r: (r["fid"] is None, r["fid"] if r["fid"] is not None else 0))
    md = [
        "# Leaderboard", "",
        "| rank | run_id | model | fid | sharpness | diversity | nn_min | epochs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows_sorted, 1):
        def f(x, fmt=".3f"):
            return format(x, fmt) if isinstance(x, (int, float)) else "-"
        md.append(
            f"| {i} | `{r['run_id']}` | {r['model']} | {f(r['fid'])} | "
            f"{f(r['sharpness'], '.4f')} | {f(r['diversity'])} | {f(r['nn_min'])} | {r['epochs']} |"
        )
    (REPORTS / "leaderboard.md").write_text("\n".join(md) + "\n")
    print(f"wrote leaderboard with {len(rows)} runs to {REPORTS}")


if __name__ == "__main__":
    main()
