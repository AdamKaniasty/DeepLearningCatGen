# Presentation Plan — Cat Generative Models

Three sections matching the project: dataset, what we do, experiments. Each bullet lists the artifact path that backs the slide.

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
- Eval protocol: FID(1000 gen vs 1000 ref) + per-epoch sample grids + latent interpolation
- Cluster: Lightning AI; local Mac/MPS for smoke

## 3. Experiments

### 3a. Per-model blocks (DCGAN / AAE / VQ-VAE)

- Sweep table (configs run)
  - `reports/leaderboard.csv` filtered by model
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

- FID bar chart per family (best run highlighted)
  - `presentation/figures/fid_bar.png` (built by `scripts/build_figures.py`)
- Side-by-side sample grids (DCGAN best | AAE best | VQ-VAE best)
  - `presentation/figures/compare_grid.png`
- Interpolation strips (DCGAN best, AAE best)
  - `runs/<best>/eval/interpolation.png`
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
