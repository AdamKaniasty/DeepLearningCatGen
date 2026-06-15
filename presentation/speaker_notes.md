# CatGen Presentation Speaker Notes

Use this as a preparation script, not as slide content. The goal is to know what each slide means, where each number comes from, and how to answer follow-up questions.

## Slide 1 — CatGen

### What to say

This project compares three lightweight generative models for cat image generation: DCGAN, Adversarial Autoencoder, and VQ-VAE. The task is not to build a state-of-the-art generator, but to compare model families under a constrained course-project setup.

### Terms

- **Generative model:** a model that learns to produce new samples resembling the training data.
- **Cat image generation:** samples are generated images, not class labels. This is why there is no accuracy, macro-F1, or confusion matrix.
- **DCGAN:** Deep Convolutional GAN, an adversarial generator/discriminator model.
- **AAE:** Adversarial Autoencoder, an autoencoder with adversarial latent-space regularization.
- **VQ-VAE:** Vector-Quantized VAE, an autoencoder with a discrete learned codebook.

### Likely question

**Why these three?**  
They represent three different generative strategies: direct adversarial generation, continuous-latent autoencoding, and discrete-latent autoencoding.

## Slide 2 — Goal

### What to say

The research question is: among lightweight CNN-scale models, which model produces the best cat images under a fixed GPU budget? We evaluate using FID, visual sample grids, interpolation for continuous-latent models, learning curves, and simple quality diagnostics.

### Terms

- **Lightweight model:** a relatively small custom network, not a large modern generator such as StyleGAN or diffusion.
- **Fixed GPU budget:** models were compared in a constrained setting, so the point is a fair practical comparison rather than absolute best possible generation.
- **FID:** Frechet Inception Distance, a distribution-level metric comparing generated images to real reference images in an Inception feature space. Lower is better.
- **Sample grid:** a visual panel of generated images. It helps inspect artifacts that FID may hide.
- **Latent interpolation:** generate images while smoothly moving between two latent vectors. This checks whether the latent space behaves smoothly.
- **Cats + dogs extension:** after finding the best model family, the best model was trained with mixed cats and dogs to see whether adding a related animal class helped or hurt generation.

### Likely question

**Why not use classification metrics?**  
Because this is an image generation task. There are no predicted class labels. The model outputs images, so we evaluate image distribution similarity and visual quality.

## Slide 3 — Dataset

### What to say

The main dataset is the Kaggle Cat Dataset, with about 10k available cat images. Part A uses a smaller 64x64 setup: 1500 train cats and 500 held-out reference cats. Parts B and C use 128x128 images with 3000 train cats and 1000 reference cats. Dogs are used only in the extension experiment.

### Terms

- **Train split:** images used to fit model parameters.
- **Held-out reference split:** real images not used for training, used as the reference distribution for FID.
- **FID reference cats:** the real images cleanfid compares against generated samples.
- **Resize/crop:** transform input images to the required resolution.
- **RGB:** three-channel color image.
- **Normalize to [-1, 1]:** pixel values are scaled so the networks operate on a consistent numeric range; output layers use `tanh`, which naturally produces values in this range.
- **Horizontal flip:** data augmentation that mirrors images during training, increasing variation without changing class semantics.
- **RandomResizedCrop:** augmentation used in the 128x128 phase; it crops a random region and resizes it, forcing the model to see varied framing.

### Likely question

**Can FID from 64x64 and 128x128 be directly compared?**  
No. Different resolution and different reference splits affect FID. Within each phase the comparison is meaningful; across phases it is only a rough trend.

## Slide 4 — Models

### What to say

The three models differ in how they represent the latent space and how they generate images. DCGAN maps random noise directly to images. AAE reconstructs images and regularizes a continuous latent space. VQ-VAE reconstructs through a discrete codebook.

### Terms

- **Latent variable `z`:** a lower-dimensional hidden representation used to generate or reconstruct images.
- **Continuous latent:** latent values are real numbers, usually sampled from a Gaussian.
- **Discrete codes:** latent representation is an index into a finite codebook.
- **Generator:** network mapping random latent noise to an image.
- **Discriminator:** network judging whether an image is real or generated.
- **Encoder:** network compressing an image into latent representation.
- **Decoder:** network mapping latent representation back to an image.
- **Codebook:** a learned table of embedding vectors. VQ-VAE replaces encoder outputs with nearest codebook entries.
- **Params:** approximate number of trainable parameters. This gives model capacity context.

### Likely question

**Why does VQ-VAE have fewer parameters but worse FID?**  
Parameter count is not the whole story. VQ-VAE also needs a good prior over discrete codes for generation. Here we sample codes empirically, without a learned autoregressive prior, which limits realism and diversity.

## Slide 5 — Training Objectives

### What to say

Each model optimizes a different objective. DCGAN optimizes adversarial binary cross-entropy losses. AAE optimizes reconstruction plus adversarial latent regularization. VQ-VAE optimizes reconstruction plus codebook commitment.

### DCGAN losses

- **Discriminator BCE:** binary cross-entropy with logits. The discriminator receives real images with target near 1 and fake images with target 0.
- **With logits:** the model outputs raw scores, not probabilities; `binary_cross_entropy_with_logits` combines sigmoid and BCE in a numerically stable way.
- **Real/fake logits:** discriminator raw outputs before sigmoid.
- **Generator non-saturating BCE:** the generator is trained so generated images are classified as real by the discriminator. In code: `BCEWithLogits(D(G(z)), 1)`.
- **Why non-saturating?** Original minimax GAN loss can produce weak gradients when the discriminator is strong. Non-saturating loss gives stronger generator gradients.
- **Label smoothing:** real target is set to `1 - label_smooth`, e.g. 0.9 instead of 1.0. This can reduce discriminator overconfidence.

### AAE losses

- **Reconstruction MSE:** mean squared error between input image `x` and reconstruction `x_rec`. It penalizes pixel-level differences.
- **Latent discriminator BCE:** a discriminator tries to distinguish true Gaussian samples from encoded latent vectors.
- **Encoder adversarial BCE:** the encoder tries to fool the latent discriminator, making encoded latents look like samples from the Gaussian prior.
- **Why do this?** If the latent space follows a known prior, we can sample random `z` and decode it into images.

### VQ-VAE losses

- **Reconstruction MSE:** pixel-level reconstruction loss between original and decoded image.
- **Vector quantization:** encoder output is replaced by the nearest learned codebook vector.
- **Commitment loss:** encourages encoder outputs to stay close to selected codebook vectors so code assignments are stable.
- **Perplexity:** logged diagnostic for codebook usage. Higher perplexity usually means more codes are being used.

### Likely question

**Which loss did you optimize for image quality?**  
None directly optimizes FID. Training optimizes the model losses above; FID is computed after generation as an evaluation metric.

## Slide 6 — Part A: What We Tried

### What to say

Part A is the 64x64 sweep. The purpose was model selection: try several configurations per family and identify which family is worth scaling. The sweep varied latent dimension, learning rate, codebook size, generator upsampling, and label smoothing. Then best configurations were refined for 80 epochs.

### Terms

- **Sweep:** a set of experiments over hyperparameters.
- **Hyperparameters:** settings chosen before training, such as learning rate or latent dimension.
- **Latent size:** dimensionality of random vector `z`.
- **Learning rate:** step size used by Adam optimizer.
- **Codebook size K:** number of discrete code vectors in VQ-VAE.
- **Upsample+conv generator:** generator uses nearest-neighbor upsampling followed by convolution instead of transposed convolution. This can reduce checkerboard artifacts.
- **Separate G/D learning rates:** generator and discriminator can learn at different speeds.

### Chart explanation

The bar chart shows best FID per model family in the refined 64x64 setup. Lower FID is better. DCGAN is best, but the absolute FID values are still high.

### Likely question

**Was hyperparameter optimization extensive?**  
Moderate, not exhaustive. It covered main parameters but not a full grid over all combinations.

## Slide 7 — Part A Sample Grids

### What to say

The image compares generated samples from DCGAN, AAE, and VQ-VAE at 64x64. DCGAN has the best FID among the three, AAE is smoother, and VQ-VAE has the weakest distribution match.

### Terms

- **Eval samples:** images sampled after training and used for qualitative inspection and FID.
- **FID labels under panels:** these are from the representative refined runs.
- **Blob-like textures:** generated images may have cat-like colors or local textures but do not form coherent cat shapes.

### Likely question

**Why does the VQ-VAE panel look repetitive/noisy?**  
Generation samples discrete codes using empirical frequencies rather than modeling spatial structure. That weak sampling method can produce poor global image structure.

## Slide 8 — Part A Outcome

### What to say

The 64x64 phase was useful for ranking model families but not enough for convincing image quality. DCGAN had the best FID, AAE was smoother but less realistic, and VQ-VAE had the weakest distribution match.

### Terms

- **Distribution match:** how close the generated-image distribution is to the real reference distribution.
- **Smoother outputs:** common in MSE-trained autoencoders because pixel-wise averaging tends to blur high-frequency details.
- **Most structured:** DCGAN samples show more image-like shapes than the others, even though they are still noisy.

### Likely question

**Why continue after bad 64x64 visuals?**  
Because Part A identified DCGAN as the best family and showed that higher resolution/more data should be tested before drawing final conclusions.

## Slide 9 — Part B: 128x128, 3000 Cats

### What to say

In Part B, all three families are retrained at 128x128 with 3000 training cats. Evaluation uses FID against 1000 held-out reference cats. This is the main higher-resolution comparison.

### Terms

- **128x128:** generated images have four times as many pixels as 64x64. This allows more detail but makes learning harder.
- **3000 train cats:** larger split than Part A, giving more variation.
- **Held-out reference cats:** real images not used for training; used for FID.

### Chart explanation

DCGAN improves clearly and reaches FID 232.964. AAE also improves relative to its Part A refined result. VQ-VAE performs worse in this representative 128x128 run, which suggests the current VQ-VAE sampling strategy does not scale well.

### Likely question

**Does 128x128 always improve FID?**  
No. DCGAN improved, AAE improved modestly, and VQ-VAE got worse in this setup. Resolution alone is not enough; the model and sampling method matter.

## Slide 10 — Part B Sample Grids

### What to say

At 128x128, DCGAN produces the most recognizable cat-like images and has the best FID. AAE remains smoother and less realistic. VQ-VAE has high FID and low diversity.

### Terms

- **Recognizable cat-like fragments:** samples contain heads, fur patterns, or body-like structures, even if distorted.
- **Artifacts:** unnatural distortions, broken faces, inconsistent texture, or non-cat blobs.

### Likely question

**Why is VQ-VAE FID so high?**  
The VQ-VAE reconstruction model alone is not enough for high-quality generation. It needs a stronger prior over code sequences. Empirical code-frequency sampling ignores spatial dependencies.

## Slide 11 — Part B DCGAN @128x128

### What to say

This is the best 128x128 result: DCGAN with FID 232.964. The sample panel shows better cat-like structure than 64x64, but still not realistic photographs. It is the best model in this project, not a solved generation problem.

### Terms

- **Run ID:** `dcgan_e9574605_42` identifies the exact run/config/seed artifact.
- **Best result:** best among completed project runs, not state-of-the-art.
- **Distorted faces/textures:** failures visible even when FID is best.

### Likely question

**Why is FID still high if this is the best model?**  
The model is small, the dataset split is limited, and generated images still have many artifacts. FID rewards distribution similarity, and these samples are still far from real cat photos.

## Slide 12 — Latent Interpolation

### What to say

Interpolation shows generated images along a path between latent vectors. DCGAN and AAE have continuous latent spaces, so interpolation is meaningful. VQ-VAE has discrete codes and no latent interpolation implemented in this pipeline.

### Terms

- **Latent interpolation:** choose two latent vectors and generate intermediate points between them.
- **Smooth latent space:** small latent changes produce gradual image changes.
- **Not a substitute for FID:** interpolation can look smooth even when individual samples are unrealistic.

### Likely question

**Why no VQ-VAE interpolation?**  
Our VQ-VAE generation uses discrete code sampling. Interpolating discrete code indices is not directly comparable to continuous `z` interpolation, and we did not implement codebook embedding interpolation as a separate analysis.

## Slide 13 — Cats vs Dogs

### What to say

This extension tests whether training DCGAN with cats plus dogs helps or hurts. It uses the same architecture and compares cats-only against mixed training. The question is whether more animal data improves robustness or dilutes the target cat distribution.

### Terms

- **Mixed training:** training data contains both cats and dogs.
- **Target distribution:** the desired output distribution; here, primarily cats.
- **Dilution:** adding dogs may cause the generator to spend capacity modeling dogs or ambiguous animal features instead of cats.

### Likely question

**Why only run the extension for DCGAN?**  
Because DCGAN was the best-performing family at 128x128. Running all extensions for all models would increase compute without clear benefit.

## Slide 14 — Part C: G/D Rebalance

### What to say

Part C investigates a training-dynamics issue in DCGAN. The generator loss plateau suggested the discriminator may be too strong. We tried increasing generator learning rate, decreasing discriminator learning rate, and reducing label smoothing.

### Terms

- **G/D:** generator/discriminator.
- **Generator-loss plateau:** generator loss stops improving, suggesting the generator is not making progress.
- **Discriminator winning:** discriminator distinguishes real from fake too easily, giving the generator an unfavorable training signal.
- **`lr` (G):** generator learning rate.
- **`lr_d`:** discriminator learning rate.
- **Label smoothing:** lowers target for real labels, e.g. from 1.0 to 0.9 or 0.95, to reduce discriminator confidence.

### Chart explanation

The FID worsened from 232.964 to 243.147. So this simple rebalance did not improve image distribution quality.

### Likely question

**Why did increasing generator LR not help?**  
GAN training is unstable. A simple LR change can worsen dynamics, overshoot, or create worse samples even if the motivation is reasonable.

## Slide 15 — Baseline vs Rebalance Samples

### What to say

This visual comparison supports the FID result. The rebalanced DCGAN does not visibly improve over the baseline. The baseline remains the better 128x128 DCGAN result.

### Terms

- **Baseline DCGAN:** Part B best run, `dcgan_e9574605_42`.
- **Rebalanced DCGAN:** Part C run, `dcgan_30f57e20_42`.
- **Eval samples:** generated samples produced after training for comparison.

### Likely question

**Could Part C still be useful if FID worsened?**  
Yes. It tests a plausible diagnosis and shows that this simple modification was insufficient. Negative results are still informative.

## Slide 16 — Learning Curves: DCGAN

### What to say

This slide shows DCGAN training diagnostics. In GANs, losses are not direct image-quality metrics. The generator and discriminator are in a game, so a loss curve can look unstable or plateau even while sample quality changes.

### Terms

- **`loss_d`:** discriminator loss. It combines real-image BCE and fake-image BCE.
- **`loss_g`:** generator loss. It is BCE where generated images are targeted as real.
- **`sample_std`:** standard deviation of fixed generated samples. Very low values can indicate mode collapse.
- **Mode collapse:** generator produces too few distinct outputs.
- **Diagnostic, not objective quality:** GAN losses do not monotonically correlate with FID.

### Likely question

**Why not early stop on DCGAN loss?**  
GAN losses are not reliable validation objectives. A lower generator loss does not always mean better images, and discriminator/generator losses depend on each other.

## Slide 17 — Learning Curves: AAE

### What to say

AAE curves include reconstruction loss and adversarial latent losses. Reconstruction measures how well the autoencoder copies input images. The adversarial terms measure whether the encoded latent distribution is being pushed toward the Gaussian prior.

### Terms

- **Reconstruction loss:** MSE between original and reconstructed image.
- **`g_adv`:** encoder adversarial loss; encoder tries to fool the latent discriminator.
- **`d_adv`:** latent discriminator loss; discriminator tries to distinguish real Gaussian `z` from encoded `z`.
- **Gaussian prior:** standard normal distribution used for sampling latent vectors.
- **Sample standard deviation:** diversity proxy for decoded fixed random latents.

### Likely question

**Why are AAE samples blurrier?**  
MSE reconstruction encourages average pixel predictions. That tends to smooth high-frequency texture, especially when the model is uncertain.

## Slide 18 — Learning Curves: VQ-VAE

### What to say

VQ-VAE curves track reconstruction loss, commitment loss, perplexity, and codebook usage. The model reconstructs through a discrete bottleneck. Good reconstructions do not automatically imply good free generation, because generation also requires a good prior over code patterns.

### Terms

- **Reconstruction loss:** MSE between input and decoded output.
- **Commitment loss:** penalizes encoder outputs for moving too far from selected codebook vectors.
- **Perplexity:** approximate effective number of codebook entries being used.
- **Codebook usage:** how many discrete codes are used versus dead.
- **Dead codes:** codebook vectors rarely or never selected.
- **Empirical code sampling:** sample code indices according to observed training frequencies; this does not model spatial dependencies.

### Likely question

**Why can VQ-VAE reconstruct but generate badly?**  
Reconstruction uses codes produced by real images. Generation samples codes without a learned spatial prior, so the decoded code grid may not correspond to realistic image structure.

## Slide 19 — Quality Diagnostics @128x128

### What to say

These diagnostics are from `eval/quality.json` for one run per model. The mean and standard deviation are over generated samples/features, not repeated experimental seeds. They support the FID and visual analysis but should not be treated as statistical confidence intervals across runs.

### Metrics

- **Sharpness sample μ±σ:** computed as Laplacian variance for each generated image, then mean and standard deviation over 1000 generated images.
- **Laplacian variance:** apply a Laplacian edge filter to a grayscale image and compute variance of the filter response. More edges/high-frequency detail usually gives higher variance.
- **What sharpness measures:** local edge/detail strength.
- **What sharpness does not measure:** semantic correctness. A noisy image can be sharp; a realistic soft image can have lower sharpness.
- **Diversity sample μ±σ:** compute Inception features for generated samples, calculate pairwise distances, then report mean and std of those distances.
- **Inception features:** vector representations from a pretrained Inception network, used as a semantic feature space.
- **Higher diversity:** generated samples are more spread out in feature space.
- **Low diversity:** samples are more similar, suggesting repeated modes or poor variation.
- **NN mean:** for each generated image, find nearest training-reference image in Inception feature space, then average those nearest distances.
- **Purpose of NN mean:** rough memorization check. If distances were extremely small, generated images might be near copies.
- **Limitation of NN mean:** it does not prove absence of memorization; it is only a first automated check.

### Values to know

- DCGAN: best FID 232.964 and highest diversity 18.493±1.736.
- AAE: FID 289.050, lower sharpness 0.0087±0.0024, consistent with smoother outputs.
- VQ-VAE: worst FID 489.819, low diversity 9.590±1.800, but sharpness 0.0178±0.0010 is high because sharpness can also reflect local edges/noise.

### Likely question

**Are these means/stds repeated experiments?**  
No. They are single-run diagnostics over generated samples/features. We did not run repeated seeded trials for confidence intervals.

## Slide 20 — Conclusions

### What to say

Across completed experiments, DCGAN is the best model family. At 64x64, DCGAN had the best FID but poor visuals. At 128x128 with more data, DCGAN improved to FID 232.964. The G/D rebalance did not help; FID worsened to 243.147.

### Terms

- **Best cats FID:** lowest FID among cats-only runs in that phase.
- **Fair comparison:** same splits and evaluation setup within each phase.
- **Simple LR/smoothing tweak:** Part C changed only learning rates and label smoothing, not architecture or dataset.

### Likely question

**What is the main conclusion in one sentence?**  
Under this lightweight setup, DCGAN is the strongest of the three tested families, but output quality is still far from realistic.

## Slide 21 — Limitations

### What to say

The project has important limitations. The networks are small, FID is not comparable across different resolutions/splits, no human study was performed, VQ-VAE interpolation was not implemented, and repeated seeded trials were not performed.

### Terms

- **StyleGAN/diffusion:** stronger modern generative model families not used here.
- **FID-only limitation:** FID is useful but incomplete and can miss visual artifacts.
- **No confidence interval:** because there were no repeated seeds, we cannot report mean±std FID across independent training runs.
- **Part A smaller splits:** Part A used less data, so it is not directly equivalent to the 128x128 phase.

### Likely question

**What would improve this project most?**  
For DCGAN: more systematic hyperparameter search and stronger architecture. For VQ-VAE: train a proper prior over discrete codes. For evaluation: repeated seeds and human/qualitative scoring.

## Slide 22 — Thank You

### What to say

End by inviting questions and pointing to the repository. If asked about reproducibility, mention run IDs, manifests, `metrics.csv`, `events.jsonl`, `fid.json`, `quality.json`, generated sample grids, and curves are stored under each run directory.

### Likely question

**Where can I verify the results?**  
The run artifacts are under `runs/<run_id>/eval/`, and the aggregate leaderboard is in `reports/leaderboard.md` and `reports/leaderboard.csv`.

