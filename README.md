# Cat Generative Models — DL Project III

Lightweight gen models (DCGAN, AAE, VQ-VAE) on the Cat Dataset, compared via FID + qualitative + latent interpolation. Cats-vs-dogs extension as exploratory.

Stack: PyTorch Lightning, CSV+JSONL logs, Python scripts only.
Local Mac/MPS for smoke tests; Lightning AI clusters for full training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Artifact contract

Every train/eval run writes to `runs/<run_id>/`:

- `manifest.json` — config, seed, git sha, dataset hash, env, timestamps
- `metrics.csv` — flat schema `step,epoch,split,name,value`
- `events.jsonl` — structured events (`epoch_end`, `mode_collapse_warning`, `fid_computed`, ...)
- `samples/epoch_XXX.png` — sample grid every K epochs
- `checkpoints/{last,best}.ckpt`
- `eval/fid.json`, `eval/samples_grid.png`, `eval/diagnostics.json`, `eval/interpolation.png`
- `summary.md` — auto TL;DR

`run_id = <model>_<short_config_hash>_<seed>` so reruns are idempotent and agents can identify configs from paths alone.

Cross-run rollup: `reports/leaderboard.{csv,md}`.

## Layout

```
data/{raw,splits}/     # dataset + fixed split files (gitignored raw)
src/catgen/            # package
runs/<run_id>/         # per-run artifacts
reports/               # cross-run rollups
scripts/               # data prep + orchestration shells
```

## Usage

```bash
python -m catgen.train --config src/catgen/configs/dcgan_a.yaml --device auto
python -m catgen.sample --run-id <run_id> --n 1000
python -m catgen.eval_fid --run-id <run_id>
python -m catgen.leaderboard
```
