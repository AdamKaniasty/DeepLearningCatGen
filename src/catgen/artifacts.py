"""Per-run artifact writers. Plain functions, no abstractions."""
from __future__ import annotations

import csv
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def run_id(model: str, config: dict, seed: int) -> str:
    h = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{model}_{h}_{seed}"


def run_dir(rid: str) -> Path:
    d = RUNS / rid
    for sub in ("samples", "checkpoints", "eval"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def write_manifest(d: Path, *, model: str, config: dict, seed: int, dataset_hash: str | None = None) -> None:
    (d / "manifest.json").write_text(json.dumps({
        "run_id": d.name,
        "model": model,
        "config": config,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "running",
    }, indent=2))


def mark_done(d: Path, status: str = "done", **extra) -> None:
    p = d / "manifest.json"
    m = json.loads(p.read_text())
    m["status"] = status
    m["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    m.update(extra)
    p.write_text(json.dumps(m, indent=2))


def log_metrics(d: Path, *, step: int, epoch: int, split: str, values: dict[str, float]) -> None:
    p = d / "metrics.csv"
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "epoch", "split", "name", "value"])
        for k, v in values.items():
            w.writerow([step, epoch, split, k, float(v)])


def log_event(d: Path, kind: str, **payload) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **payload}
    with (d / "events.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def write_summary(d: Path, text: str) -> None:
    (d / "summary.md").write_text(text)


def is_done(rid: str) -> bool:
    p = RUNS / rid / "manifest.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "done"
    except Exception:
        return False


def _smoke() -> Path:
    d = run_dir("_smoke")
    write_manifest(d, model="smoke", config={"hello": "world"}, seed=0, dataset_hash="deadbeef")
    log_metrics(d, step=0, epoch=0, split="train", values={"loss": 1.23, "acc": 0.5})
    log_metrics(d, step=1, epoch=0, split="train", values={"loss": 1.0, "acc": 0.6})
    log_event(d, "epoch_end", epoch=0, loss=1.0)
    write_summary(d, "# Run _smoke\n\n- model: smoke\n- status: smoke-test\n")
    mark_done(d)
    return d


if __name__ == "__main__":
    print(_smoke())
