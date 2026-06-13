# Presentation Plan — Cat Generative Models

**Slide deck (MARP):** [`slides.md`](slides.md) · export: `npm run presentation:pdf` (see [`README.md`](README.md))

Three sections matching the project: dataset, what we do, experiments. Each bullet lists the artifact path that backs the slide.

## Training protocol (three parts for the story)

**Part A — Sweep + refine (64×64, did not reach good visual quality)**  
Fair comparison of three lightweight models; limited by resolution and eden split sizes (1500 train / 500 ref).

1. **Sweep** — Grid over DCGAN / AAE / VQ-VAE (`scripts/gen_configs.py`), plus cats+dogs **DCGAN** extension. Figures: `presentation/figures/` with `PRESENTATION_PHASE=sweep` (default).
2. **Refine** — 80-epoch rerun with DCGAN upsample-conv + training tweaks. Figures: same folder, `PRESENTATION_PHASE=refine`.

Use Part A for methodology and “what we tried”; samples stay blobby at 64×64 (FID ~275–420).

**Part B — High-res follow-up (128×128, 3000 cats, scope splits)**  
Shows that **resolution + data** matter more than small 64×64 tweaks.

- Train all three families at **128×128**, `train_3000.txt`, `RandomResizedCrop` aug (`scripts/slurm/run_phase128.sh`).
- **Cats vs dogs** only for the **best** of the three by cats-only FID (mixed `mixed_train_800`, same as extension).
- Figures: `presentation/figures/phase128/` via `PRESENTATION_PHASE=phase128` (`fid_bar.png`, `compare_grid.png`, `interp_dcgan.png`, `interp_aae.png`, `ext_compare.png`).
- Example DCGAN run: `dcgan_e9574605_42` (FID ~233 at 128×128). Leaderboard summary: `reports/phase128_best.json`.

**Part C — DCGAN G/D rebalance @128 (optional follow-up)**  
Same data as Part B (`train_3000`, 128×128); tune generator vs discriminator balance after `loss_g` plateau / D-winning curves in Phase B.

| | Phase B (`dcgan_e9574605_42`) | Part C (`dcgan_30f57e20_42`) |
|--|-------------------------------|------------------------------|
| `lr` (G) | 1e-4 | **2e-4** |
| `lr_d` | 2e-4 | **1e-4** |
| `label_smooth` | 0.1 | **0.05** |

Run on eden: `sbatch scripts/slurm/catgen_partc_tesla.sbatch` · figures: `PRESENTATION_PHASE=partc python scripts/build_figures.py` → `presentation/figures/partc/`

**Slide order suggestion:** Dataset → Part A (sweep/refine table + weak 64×64 grids) → Part B (128×128 compare + samples + interpolation + extension for best model).

## Cluster (eden) actual sizes

Scope.pdf lists 3000/1000 cats and 1000+1000 mixed; the eden pipeline uses smaller splits from [`scripts/slurm/sizes.env`](../scripts/slurm/sizes.env) (defaults: **1500** train cats, **500** FID ref, **400+400** mixed train, **200+200** mixed ref, **500** FID samples per cat run). Use authoritative counts in **`data/splits/manifest.json`** on the cluster after `prepare_data.py`. Presentation figures are built by `scripts/slurm/run_finalize.sh` after resume + extension jobs. VQ-VAE codebook sweep on P100: **K=128, 256** (K=512 dropped — OOM on 16GB GPUs).

## 1. Dataset

- Source + license: Cat Dataset (Kaggle, 9 993 imgs), Dogs vs Cats
- Counts: train 3 000 / fid_ref 1 000 / reserve 5 993 / mixed 1k cats + 1k dogs
  - `data/splits/manifest.json`
- Preprocessing: resize 64x64, normalize [-1, 1], hflip aug (train only)
- Determinism: seed=42, per-split SHA1 hash recorded
- Raw image grid (16 cats + 8 dogs)
  - `presentation/figures/dataset_preview.png` (built by `scripts/preview_dataset.py`)

## 2. What we do

- Goal: compare 3 lightweight gen models on cats — fair comparison over absolute quality
- Models: DCGAN-64, Conv-AAE-64, VQ-VAE-64 (EMA codebook)
- Param counts per model
  - `runs/<rid>/manifest.json` field `n_params`
- Training stack: PyTorch Lightning, CSV+JSONL artifacts, run_id = `<model>_<8hex>_<seed>`
- Eval protocol: FID(gen vs ref split) + per-epoch sample grids + latent interpolation (DCGAN/AAE)
- Two-phase training: sweep → pick best → refine (see **Training protocol** above)
- Cluster: Lightning AI; local Mac/MPS for smoke

## 3. Experiments

### 3a. Per-model blocks (DCGAN / AAE / VQ-VAE)

- **Phase 1 — sweep** table (all grid configs)
  - `reports/leaderboard.csv` filtered by model, exclude `refine` tag
- **Phase 2 — refinement** (optional slide): what changed vs sweep best (epochs, generator, lr)
  - `runs/<refine_rid>/manifest.json` (`tags: [refine]`)
- Training curves (loss_d / loss_g / recon / perplexity / codes_used / sample_std)
  - `runs/<rid>/metrics.csv`
  - `runs/<rid>/eval/curves.png` (built by `scripts/plot_metrics.py`)
- Sample grid progression: epoch 0 / mid / final
  - `runs/<rid>/samples/epoch_*.png`
- Final sample grid (best config)
  - `runs/<best>/samples/epoch_<last>.png`
- Per-config FID + quality metrics
  - `runs/<rid>/eval/fid.json`
  - `runs/<rid>/eval/quality.json` (sharpness via Laplacian variance, diversity via mean pairwise Inception distance, nearest-neighbor distance to training set)
  - `runs/<rid>/eval/sharpness.npy`, `runs/<rid>/eval/nn_distances.npy` (per-sample arrays for histograms)
- Per-epoch checkpoint trace (DCGAN only)
  - `runs/<rid>/checkpoints/epoch_XXX.ckpt` saved every `save_every` epochs — supports "earlier-checkpoint rescue" if late epochs collapse
- Model-specific diagnostics:
  - DCGAN: `events.jsonl` mode_collapse_warning, sample_std curve, periodic checkpoints
  - AAE: reconstruction grid `runs/<rid>/samples/recon_epoch_*.png`, interpolation `runs/<rid>/eval/interpolation.png`
  - VQ-VAE: codebook histogram `runs/<rid>/eval/codebook_hist.png`, dead-codes/perplexity curves

### 3b. Cross-model comparison

- FID bar chart per family (best run in active phase: sweep default, refine after phase 2)
  - `presentation/figures/fid_bar.png` (`scripts/build_figures.py`, `PRESENTATION_PHASE=refine` for final deck)
- Side-by-side sample grids (DCGAN best | AAE best | VQ-VAE best)
  - `presentation/figures/compare_grid.png`
- Interpolation strips (DCGAN best, AAE best)
  - `presentation/figures/interp_dcgan.png`, `interp_aae.png` (copied from `runs/<best>/eval/interpolation.png`)
- Discussion: sharpness, diversity, mode collapse, training stability

### 3c. Cats + Dogs extension

- Cats-only best vs cats+dogs best, side by side
  - `presentation/figures/ext_compare.png`
- Mixed FID vs cats-only FID
  - `runs/<rid>/eval/fid.json` for both
- Visual: distinct classes vs blends

## Required artifacts checklist

| Artifact | Built by | Status |
|---|---|---|
| `data/splits/manifest.json` | `scripts/prepare_data.py` | exists |
| `presentation/figures/dataset_preview.png` | `scripts/preview_dataset.py` | new |
| `runs/<rid>/manifest.json` (incl. n_params) | `catgen.train` | extend |
| `runs/<rid>/metrics.csv` | training | exists |
| `runs/<rid>/events.jsonl` | training | exists |
| `runs/<rid>/samples/epoch_*.png` | training | exists |
| `runs/<rid>/samples/recon_epoch_*.png` (AAE only) | AAE LM | new |
| `runs/<rid>/eval/curves.png` | `scripts/plot_metrics.py` | new |
| `runs/<rid>/eval/codebook_hist.png` + `codebook_counts.npy` (VQ only) | VQ LM | new |
| `runs/<rid>/eval/fid.json` | `catgen.eval_fid` | exists |
| `runs/<rid>/eval/quality.json` (sharpness, diversity, NN-to-train) | `catgen.eval_quality` | exists |
| `runs/<rid>/checkpoints/epoch_XXX.ckpt` (DCGAN) | trainer `ModelCheckpoint` (save_every) | exists |
| `runs/<rid>/eval/interpolation.png` | `catgen.interpolate` | exists |
| `reports/leaderboard.{csv,md}` | `catgen.leaderboard` | exists |
| `reports/report_bundle.md` | `scripts/build_report.py` | exists |
| `presentation/figures/fid_bar.png` | `scripts/build_figures.py` | new |
| `presentation/figures/compare_grid.png` | `scripts/build_figures.py` | new |
| `presentation/figures/ext_compare.png` | `scripts/build_figures.py` | new |
| `presentation/figures/interp_dcgan.png` | `scripts/build_figures.py` | refine phase |
| `presentation/figures/interp_aae.png` | `scripts/build_figures.py` | refine phase |
| Refine runs (×4: dcgan, aae, vqvae, dcgan_ext) | `scripts/slurm/run_refine.sh` | after phase 2 job |
| DCGAN 128×128, 3000 cats | `scripts/slurm/run_dcgan128.sh` | optional quality pass |
