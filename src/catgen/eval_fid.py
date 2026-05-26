from __future__ import annotations

import argparse
import json
import os

from cleanfid import fid

from catgen import artifacts
from catgen.data import RAW, read_split


def materialize_ref(split: str):
    out = artifacts.ROOT / "data" / "splits" / f".{split}.imgs"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*"):
        if p.is_symlink():
            p.unlink()
    for p in read_split(split):
        (out / p.name).symlink_to(p.resolve())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ref-split", default="fid_ref_1000.txt")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    d = artifacts.RUNS / args.run_id
    gen_dir = d / "eval" / "samples"
    if not any(gen_dir.glob("*.png")):
        raise SystemExit(f"no samples in {gen_dir}; run catgen.sample first")

    ref_dir = materialize_ref(args.ref_split)
    score = fid.compute_fid(
        str(ref_dir),
        str(gen_dir),
        mode="clean",
        num_workers=args.num_workers,
        device="cpu",
    )
    result = {
        "fid": float(score),
        "ref_split": args.ref_split,
        "n_ref": sum(1 for _ in ref_dir.iterdir()),
        "n_gen": sum(1 for _ in gen_dir.glob("*.png")),
    }
    (d / "eval" / "fid.json").write_text(json.dumps(result, indent=2))
    artifacts.log_event(d, "fid_computed", **result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
