# CatGen presentation (MARP)

~12-minute deck (**15 slides**): [`slides.md`](slides.md) → PDF/HTML. One main figure per slide where possible.

## Preview

1. Install [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
2. Open `slides.md` and use the preview pane

## Pre-flight (figures + cluster sync)

```bash
# Pull runs, reports, and figures from eden (excludes heavy epoch checkpoints)
./scripts/rsync_from_eden.sh

# Regenerate aggregate figures from runs/ + reports/
PRESENTATION_PHASE=refine python scripts/build_figures.py
PRESENTATION_PHASE=phase128 python scripts/build_figures.py

# Optional: dataset preview from local splits
python scripts/preview_dataset.py   # → presentation/figures/dataset_preview*.png
```

Refine figures land in `presentation/figures/`; phase128 in `presentation/figures/phase128/`. These paths are **not** committed to git — only `presentation/assets/` is staged for MARP.

## Build assets & export

```bash
bash scripts/stage_presentation_assets.sh   # reads runs.yaml → assets/
python3 scripts/validate_presentation_assets.py

npm install   # once: @marp-team/marp-cli
npm run presentation:pdf    # → presentation/catgen.pdf
npm run presentation:html   # → presentation/catgen.html
```

## Canonical runs

See [`runs.yaml`](runs.yaml) for `run_id` → artifact mapping.

**If phase128 job failed (VQ OOM / resume):** on eden, `sbatch scripts/slurm/catgen_phase128_resume_tesla.sbatch` (see `scripts/slurm/run_phase128_resume.sh`).

**After phase128 job completes on eden:**

1. `./scripts/rsync_from_eden.sh`
2. Copy FID / run ids from `reports/phase128_best.json` and `reports/leaderboard.md` into `runs.yaml` (`aae_cats`, `vqvae_cats`, `dcgan_mixed`, `fid.phase128.*`)
3. Point `assets.part128.interp_aae` and `ext_compare` at phase128 outputs (remove refine fallbacks)
4. `PRESENTATION_PHASE=phase128 python scripts/build_figures.py`
5. `bash scripts/stage_presentation_assets.sh` && `npm run presentation:pdf`

**Current placeholders:** `interp_aae` and `ext_compare` for slide 8 may use refine @64 until mixed extension and AAE eval @128 exist locally.

## Optional appendix

Not presented live; add `slides_appendix.md` for curves/codebook if needed.
