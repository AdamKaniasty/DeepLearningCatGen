---
marp: true
theme: catgen
paginate: true
footer: "CatGen — DL Project III"
---

<!-- _class: lead -->

# CatGen
## Comparing DCGAN, AAE, and VQ-VAE on cat image generation

**Your Name** · Deep Learning Project III

---

## Goal

**Question:** Which lightweight generative model produces the best cat images under a fixed GPU budget?

- Three families: **DCGAN**, **Adversarial Autoencoder (AAE)**, **VQ-VAE**
- Metrics: **FID** + sample grids + latent interpolation (DCGAN/AAE)
- Extension: **cats + dogs** training for the **best** model only (128×128 phase)

---

<!-- _class: content -->

## Dataset

- **Cats:** Kaggle Cat Dataset (~10k images)
- **Dogs:** Dogs vs Cats (extension only)
- **Train / FID ref:** 3000 / 1000 cats (128×128); 1500 / 500 for 64×64 sweep on cluster
- **Preprocessing:** resize or crop → RGB → normalize to [-1, 1]; horizontal flip (train)

![h:200](assets/dataset/dataset_preview.png)

*Training cats after preprocessing (use Kaggle preview after cluster rsync)*

---

## Models (64×128 capable)

| Model | Idea | Latent | Params (order) |
|-------|------|--------|----------------|
| **DCGAN** | Generator + discriminator | Continuous **z** | ~9.6M |
| **AAE** | Conv AE + **adversarial** latent D | Continuous **z** | ~7.7M |
| **VQ-VAE** | Encoder → **codebook** → decoder | Discrete codes | ~0.5M |

Same stack: PyTorch Lightning, reproducible `run_id`, early stopping on AAE/VQ (`recon`).

---

<!-- _class: content -->

## Part A — What we tried (64×64)

- **Sweep:** ~10 configs per family (z, lr, codebook K) on **1500** train cats
- **Refine:** 80 epochs, DCGAN **upsample+conv** G, label smoothing, separate G/D LR

![chart](assets/part64/fid_bar.png)

---

<!-- _class: figure -->

## Part A — Sample grids (64×64, eval)

![w:980](assets/part64/compare_grid.png)

**DCGAN** · **AAE** · **VQ-VAE** (left → right)

---

<!-- _class: content -->

## Part A — Outcome

- Best cats-only FID still **~275–420** (DCGAN lowest)
- **Refinement did not fix visuals** — samples remain blob-like textures

![panel](assets/part64/dcgan_samples.png)

---

<!-- _class: content -->

## Part B — 128×128, 3000 cats

- Retrain all three at **128×128** on **3000** train cats (`RandomResizedCrop` aug)
- FID vs **1000** reference cats

![chart](assets/part128/fid_bar.png)

---

<!-- _class: figure -->

## Part B — Sample grids (128×128, eval)

![w:980](assets/part128/compare_grid.png)

**DCGAN** · **AAE** · **VQ-VAE** (left → right)

---

<!-- _class: content -->

## Part B — DCGAN @ 128×128

**`dcgan_e9574605_42`** — FID **~233** (best so far)

![hero](assets/part128/hero_dcgan.png)

---

<!-- _class: pair -->

## Part B — Latent interpolation

<div class="interp">

![h:200](assets/part128/interp_dcgan.png)

**DCGAN @ 128×128**

</div>

<div class="interp">

![h:200](assets/part128/interp_aae.png)

**AAE @ 128×128**

</div>

---

<!-- _class: figure -->

## Part B — Cats vs dogs (best model)

<p class="sub">Same architecture; cats-only vs mixed training (DCGAN extension)</p>

![w:1050](assets/part128/ext_compare.png)

---

<!-- _class: content -->

## Part C — G/D rebalance @128

Phase B DCGAN: **`loss_g` plateau**, discriminator winning → try rebalancing G vs D (same 3k cats, 80 epochs)

| | Phase B | Part C |
|---|---------|--------|
| `lr` (G) | 1e-4 | **2e-4** |
| `lr_d` | 2e-4 | **1e-4** |
| `label_smooth` | 0.1 | **0.05** |

![chart h:220](assets/partc/fid_bar.png)

**FID: 233 → 243** — rebalance did **not** improve on Phase B

---

<!-- _class: figure -->

## Part C — Baseline vs rebalance (eval samples)

![w:1050](assets/partc/gd_compare.png)

`dcgan_e9574605_42` (Phase B) · `dcgan_30f57e20_42` (Part C)

---

## Conclusions

| Phase | Best model | Best cats FID |
|-------|------------|---------------|
| 64×64 refine | DCGAN | ~287 |
| 128×128 | DCGAN | **~233** |
| 128×128 G/D rebalance | DCGAN | ~243 (worse) |

1. **DCGAN** wins across phases (AAE/VQ-VAE higher FID)
2. **64×64 sweep + refine** = fair comparison, **not** good visuals
3. **128×128 + more data** = clear improvement
4. **G/D rebalance** did not beat Phase B — simple LR/smoothing tweak insufficient

---

## Limitations

- Small **custom** nets — not StyleGAN / diffusion
- **FID not comparable** across 64 vs 128 (different resolution & splits)
- **FID-only** — no human study; **VQ-VAE** has no latent interpolation in our pipeline
- Cats-vs-dogs extension only for **best** model at 128; cluster used smaller splits in Part A

---

<!-- _class: lead -->

# Thank you

**Questions?**

[github.com/AdamKaniasty/DeepLearningCatGen](https://github.com/AdamKaniasty/DeepLearningCatGen)
