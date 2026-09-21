# Domain Adaptation via Doubly-Anchored DRO

These codes reproduce the figures and results included in the paper Kepplinger, D and Vidyashankar, A.N (2026+). Domain Adaptation with Target Information via Doubly-Anchored Distributionally Robust Optimization.

The Fashion-MNIST results and figures are produced by running the two Python scripts below to generate raw CSV output, then rendering the two Quarto documents to summarize/plot that output.

## Python requirements

The Python scripts were run with Python 3.12.
The following versions are recorded from the cluster environment used for the numerical studies:

| Package | Version |
| --- | --- |
| `numpy` | 2.4.4 |
| `scipy` | 1.18.0 |
| `pandas` | 3.0.5 |
| `scikit-learn` | 1.9.0 |
| `matplotlib` | 3.11.1 |
| `torch` | 2.12.1+cu126 |
| `torchvision` | 0.27.1+cu126 |
| `adapt` | 0.4.5 |
| `scikeras` | 0.13.0 |
| `tensorflow` | 2.21.0 |

The core scripts require `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`, and `torchvision`.
The `adapt`, `scikeras`, and `tensorflow` packages are also needed by the KLIEP and ULSIF reweighting paths in `appl_fashion_mnist.py`.
Install a PyTorch and torchvision build compatible with the cluster's CUDA installation; the `+cu126` suffix above identifies the build used there rather than
a universal pip requirement.

## Training the autoencoder

The autoencoder used for the Pullover-versus-Coat experiments was trained with:

```bash
python appl_fashion_mnist-train_ae.py Pullover Coat 10 results/ae_trained_pullover_coat_10.pt
```

### 1. `appl_fashion_mnist.py` — replication runs

Fits the baseline and bridged/loss-adversarial NW models for one replication and appends per-model metrics (accuracy, AUC, Brier score, F1) to a CSV in `results/`:

```bash
python appl_fashion_mnist.py \
  "autoenc/ae_trained_pullover_coat_10.pt" \
  10 \
  laplace 0.15 \
  --nw_bw 10.0 \
  -b Coat \
  --start_replication 1 \
  --replications 100
```

Positional arguments are the trained autoencoder checkpoint, its latent
dimension, and the noise type/SD applied to the target domain.
`-b` selects the second Fashion-MNIST class (paired against the default `Pullover`),

### 2. `appl_fashion_mnist_visuals.py` — probability sweep and figures

#### Probability/weight sweep (`compute_probabilities`)

Computes bridge weights and Nadaraya-Watson probabilities across a grid of `alpha`/`gamma`/`nu` values and writes them to a single CSV.
The `nu` values need to specified for the KL/KL, HD/HD, and KL/HD bridge and hence the script needs to be run several times to cover the entire `nu` grid.

```bash
python appl_fashion_mnist_visuals.py \
  --ae "autoenc/ae_trained_pullover_coat_10.pt" \
  --ae_dim 10 \
  --noise_type laplace \
  --noise_sd 0.15 \
  --nw_bw 10.0 \
  -b Coat \
  --nu="0.010000,4.000001,0.010000" \
  --gamma="0" \
  --alpha="0.05:0.15:0.95,1" \
  --seed=1 \
  --out_path="results" \
  compute_probabilities
```

#### Figures 2(a) and 2(b) — sample images and latent-space shift

The `images` visualization type produces a grid comparing clean vs.
noise-degraded target samples (Figure 2(a)):

```bash
python appl_fashion_mnist_visuals.py \
  --noise_sd=0.15 \
  --ae autoenc/ae_trained_pullover_coat_10.pt \
  --ae_dim 10 \
  --num_images=4 \
  images
```

The `latent` visualization type projects the source and (clean/shifted) target
domains into a 2D discriminant subspace to illustrate the induced covariate
shift (Figure 2(b)):

```bash
python appl_fashion_mnist_visuals.py \
  --noise_sd=0.15 \
  --ae autoenc/ae_trained_pullover_coat_10.pt \
  --ae_dim 10 \
  latent
```

Both commands write their output image (`.png`/`.pdf`) to `figures/`.

### 3. `bridge-weights.qmd` — bridge weight geometry figures

Self-contained Quarto document (no dependency on the Python outputs) that
plots the KL/KL, HD/HD, and KL/HD bridge weight functions as a function of the
likelihood ratio and of `alpha`.
Render with:

```bash
quarto render bridge-weights.qmd
```

### 4. `appl_fashion_results.qmd` — summary figures from the sweep output

Reads the CSV outputs of both Python scripts from `results/` (the `compute_probabilities` sweep from `appl_fashion_mnist_visuals.py` and accuracy/AUC/Brier/F1 results from `appl_fashion_mnist.py`), and produces the sensitivity-analysis and target-accuracy summary figures used in the paper.
Render with:

```bash
quarto render appl_fashion_results.qmd
```

## Other files

- `bridged_nw.py` contains the weighted Nadaraya-Watson estimator, likelihood-ratio estimation, bridge weights, and loss-adversarial fitting routines.
- `fashion_mnist_helper.py` loads and filters Fashion-MNIST and defines the `AutoencoderFMNIST` model.
- `da_image_degeneration.py` functions to add Gaussian or Laplace noise to target-domain images.
- `plotting-utils.R` contains shared R plotting colors and themes used by both Quarto documents.
- `autoenc/` stores trained autoencoder checkpoints; `results/` and `figures/` contain generated tables and plots.
- `renv.lock`, `renv/activate.R`, and `renv/settings.json` record and configure the R package environment used to render the Quarto documents.