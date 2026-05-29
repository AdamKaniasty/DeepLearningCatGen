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

## Usage (local)

```bash
python -m catgen.train --config src/catgen/configs/dcgan_a.yaml --device auto
python -m catgen.sample --run-id <run_id> --n 1000
python -m catgen.eval_fid --run-id <run_id>
python -m catgen.leaderboard
```

## Lightning AI cluster

Requires in `.env`:

```
LIGHTNING_USER_ID=<uuid>
LIGHTNING_API_KEY=<uuid>
# optional overrides (auto-detected when single value exists):
# LIGHTNING_TEAMSPACE=deploy-model-project
# LIGHTNING_STUDIO_NAME=catgen
# LIGHTNING_MACHINE=T4
```

Repo must be reachable via git (default: `origin` remote).

```bash
python scripts/lightning_submit.py whoami         # auth + teamspace check
python scripts/lightning_submit.py setup          # create Studio, clone repo, install deps
python scripts/lightning_submit.py sync           # after subsequent local commits: git pull + reinstall on Studio
# (one-time, inside Studio UI: upload Cat Dataset to data/raw/cats and Dogs vs Cats to data/raw/dogs)
python scripts/lightning_submit.py data           # prepare splits on Studio
python scripts/lightning_submit.py submit dcgan   # 8 jobs (one per config)
python scripts/lightning_submit.py submit aae     # 4 jobs
python scripts/lightning_submit.py submit vqvae   # 8 jobs
python scripts/lightning_submit.py eval           # eval + figures + report on Studio
python scripts/lightning_submit.py push -m "sweep complete"
git pull origin main                              # pull runs/ + reports/ back locally
```

Dry-run any submit/eval with `--dry-run` to see the planned commands without spending credits.

## SLURM (eden)

Same pattern as Mi-Crow: **no login-node pip**. Each job runs `uv sync` on the compute node (cache in `.uv-cache/` on evafs).

```bash
ssh eden
cd /mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
git pull
bash scripts/slurm/setup_eden.sh          # clone/pull + data symlink only

sbatch scripts/slurm/catgen_sweep.sbatch  # full sweep (uv sync + train + eval)
sbatch scripts/slurm/catgen_smoke.sbatch  # quick GPU smoke (fake data)

squeue -u $USER
tail -f slurm-logs/sweep-<jobid>.out
```

Cats data defaults to `/mnt/evafs/faculty/home/kbokhan/data/cats` (symlinked by `scripts/slurm/link_data.sh`). Override with `CATGEN_DATA_CATS`.
