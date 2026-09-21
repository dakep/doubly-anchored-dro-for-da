import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from da_image_degeneration import apply_gaussian_noise, apply_laplace_noise
from fashion_mnist_helper import load_fmnist, AutoencoderFMNIST
from bridged_nw import (
  estimate_likelihood_ratio,
  fit_loss_adversarial_nw,
  get_bridge_weights,
  WeightedNadarayaWatson,
)


sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_ADV_MAX_ITER = 100
_ADV_STEP_SIZE = 0.2
_ADV_ITER_CONV_TOL = 1.0

_COLORPAL = dict(
  blue='#0072B2',
  orange='#D55E00',
  gray70='#BEBEBE',
)


class DiscriminantSubspace2D:
  """Project two populations into a two-dimensional discriminant subspace."""

  def __init__(self, ridge_alpha: float = 1e-5):
    self.ridge_alpha = ridge_alpha
    self.projection_matrix_ = None

  def _ensure_tensor(self, values):
    if isinstance(values, np.ndarray):
      return torch.tensor(values, dtype=torch.float64)
    if isinstance(values, torch.Tensor):
      return values.to(torch.float64)
    raise TypeError(f"Expected numpy array or torch Tensor, got {type(values)}")

  def fit(self, X, Y):
    X = self._ensure_tensor(X)
    Y = self._ensure_tensor(Y)
    n, p = X.shape
    m, _ = Y.shape

    mu_X = X.mean(dim=0)
    mu_Y = Y.mean(dim=0)
    within_scatter = (n - 1) * torch.cov(X.T) + (m - 1) * torch.cov(Y.T)
    ridge = self.ridge_alpha * torch.eye(p, dtype=within_scatter.dtype)
    w1 = torch.linalg.solve(within_scatter + ridge, mu_X - mu_Y)
    w1 = w1 / torch.norm(w1)

    combined = torch.cat([X, Y], dim=0)
    centered = combined - combined.mean(dim=0)
    projected = (centered @ w1).unsqueeze(1) @ w1.unsqueeze(0)
    _, _, Vh = torch.linalg.svd(centered - projected, full_matrices=False)
    w2 = Vh[0, :]

    score_1 = centered @ w1
    score_2 = centered @ w2
    beta = torch.dot(score_1, score_2) / torch.dot(score_1, score_1)
    w2 = w2 - beta * w1
    w2 = w2 / torch.norm(w2)
    self.projection_matrix_ = torch.stack([w1, w2], dim=0)
    return self

  def transform(self, values, return_numpy=False):
    if self.projection_matrix_ is None:
      raise RuntimeError("This instance is not fitted yet. Call 'fit' first.")
    values_tensor = self._ensure_tensor(values)
    transformed = values_tensor @ self.projection_matrix_.T
    if return_numpy or isinstance(values, np.ndarray):
      return transformed.cpu().numpy()
    return transformed

  def fit_transform(self, X, Y):
    self.fit(X, Y)
    return self.transform(X), self.transform(Y)


def _positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("argument must be > 0")
  return parsed


def parse_range(value):
  """Parse comma-separated values and linear ranges."""
  parsed_values = []
  try:
    for part in value.split(','):
      values = part.split(':')
      if len(values) == 1:
        parsed_values.append(float(values[0]))
      elif len(values) == 2:
        start, stop = map(float, values)
        parsed_values.extend(np.arange(start, stop + 0.1, 0.1))
      elif len(values) == 3:
        start, step, stop = map(float, values)
        parsed_values.extend(np.arange(start, stop + step, step))
      else:
        raise argparse.ArgumentTypeError(
          "Range must be 'value', 'value1,value2,...', 'min:max', or 'min:step:max'"
        )
  except ValueError as error:
    raise argparse.ArgumentTypeError("Range values must be valid numbers.") from error
  return parsed_values


def _noise_function(noise_type: str):
  return apply_gaussian_noise if noise_type == 'gaussian' else apply_laplace_noise


def _latent_embeddings(X_S, X_T, noise_type, noise_sd, autoencoder, seed):
  rng = np.random.default_rng(seed)
  noise_fn = _noise_function(noise_type)
  with torch.no_grad():
    X_T_noise = noise_fn(X_T, noise_sd=noise_sd, rng=rng)
    Z_S = autoencoder.encoder(X_S).numpy()
    Z_T_noise = autoencoder.encoder(X_T_noise).numpy()
    Z_T_clean = autoencoder.encoder(X_T).numpy()

  scaler = StandardScaler().fit(Z_S)
  return (
    scaler.transform(Z_S),
    scaler.transform(Z_T_clean),
    scaler.transform(Z_T_noise),
  )


def _standardize_source_and_targets(X_S, X_T, noise_type, noise_sd, autoencoder, seed):
  rng = np.random.default_rng(seed)
  noise_fn = _noise_function(noise_type)
  with torch.no_grad():
    X_T_noise = noise_fn(X_T, noise_sd=noise_sd, rng=rng)
    Z_S = autoencoder.encoder(X_S).numpy()
    Z_T_noise = autoencoder.encoder(X_T_noise).numpy()

  scaler = StandardScaler().fit(Z_S)
  return scaler.transform(Z_S), scaler.transform(Z_T_noise)


def plot_latent_shift_target(X_S, y_S, X_T, y_T, noise_sd,
                             autoencoder, out_file, seed=42,
                             noise_type='laplace', class_labels=('Class A', 'Class B')):
  """Plot clean source/target and noise-shifted target latent spaces."""
  Z_S, Z_T_clean, Z_T_noise = _latent_embeddings(
    X_S, X_T, noise_type, noise_sd, autoencoder, seed
  )
  subspace = DiscriminantSubspace2D()
  Z_T_noise_plot, Z_T_clean_plot = subspace.fit_transform(Z_T_noise, Z_T_clean)
  Z_S_plot = subspace.transform(Z_S)

  fig, ((ax_clean_a, ax_clean_b), (ax_noise_a, ax_noise_b)) = plt.subplots(
    2, 2, figsize=(3.3, 3.3), sharex=True, sharey=True
  )
  axes = ((ax_clean_a, ax_clean_b), (ax_noise_a, ax_noise_b))
  for class_index, class_label in enumerate(class_labels):
    source = Z_S_plot[y_S == class_index]
    clean = Z_T_clean_plot[y_T == class_index]
    noisy = Z_T_noise_plot[y_T == class_index]
    clean_ax = axes[0][class_index]
    noise_ax = axes[1][class_index]

    clean_ax.scatter(source[:, 0], source[:, 1], alpha=0.5,
                     color=_COLORPAL['gray70'], marker='o')
    clean_ax.scatter(clean[:, 0], clean[:, 1], alpha=0.5,
                     color=_COLORPAL['blue'], marker='s')
    noise_ax.scatter(source[:, 0], source[:, 1], alpha=0.5,
                     color=_COLORPAL['gray70'], marker='o')
    noise_ax.scatter(noisy[:, 0], noisy[:, 1], alpha=0.5,
                     color=_COLORPAL['orange'], marker='s')
    clean_ax.set_title(class_label, fontsize=8)
    for ax in (clean_ax, noise_ax):
      ax.set_xticks([])
      ax.set_yticks([])

  ax_clean_a.set_ylabel('Original target domain', fontsize=8)
  ax_noise_a.set_ylabel('Shifted target domain', fontsize=8)
  ax_noise_a.set_xlabel('PC1', fontsize=8)
  ax_noise_b.set_xlabel('PC1', fontsize=8)
  fig.tight_layout()
  plt.savefig(out_file, dpi=300, bbox_inches='tight')
  plt.savefig(out_file.replace('.png', '.pdf'), bbox_inches='tight')
  plt.close(fig)


def visualize_degradations(X_clean, y, out_file, noise_sd=0.3,
                           noise_type='laplace', class_labels=('Class A', 'Class B'),
                           num_images=5, seed=42, direction='h'):
  """Plot sampled clean target images beside their noise-degraded versions."""
  rng = np.random.default_rng(seed)
  if num_images % 2 != 0:
    num_images += 1
  half_n = num_images // 2
  indices_a = rng.choice(np.where(y == 0)[0], half_n, replace=False)
  indices_b = rng.choice(np.where(y == 1)[0], half_n, replace=False)
  clean_samples = torch.cat([X_clean[indices_a], X_clean[indices_b]], dim=0)

  image_sets = [clean_samples]
  image_labels = ['Original\ntarget domain']
  image_sets.append(_noise_function(noise_type)(clean_samples, noise_sd, rng))
  image_labels.append('Shifted\ntarget domain')

  if direction == 'v':
    num_rows, num_cols = num_images, len(image_sets)
  else:
    num_rows, num_cols = len(image_sets), num_images
  fig, axes = plt.subplots(num_rows, num_cols,
                           figsize=(num_cols * 1.2, num_rows * 1.2), squeeze=False)

  for row_idx in range(num_rows):
    for col_idx in range(num_cols):
      image_idx = row_idx if direction == 'v' else col_idx
      set_idx = col_idx if direction == 'v' else row_idx
      ax = axes[row_idx, col_idx]
      ax.imshow(image_sets[set_idx][image_idx].view(28, 28),
                cmap='gray', vmin=0, vmax=1)
      ax.set_xticks([])
      ax.set_yticks([])
      if direction == 'v' and row_idx == 0:
        ax.set_title(image_labels[set_idx], fontsize=8, fontweight='bold')
      elif direction == 'h' and col_idx == 0:
        ax.set_ylabel(image_labels[set_idx], fontsize=8, rotation=90, labelpad=10)

  fig.canvas.draw()
  if direction == 'v':
    first_position = axes[0, 0].get_position()
    second_position = axes[half_n, 0].get_position()
    x_label = first_position.x0 - 0.02
    fig.text(x_label, (first_position.y1 + axes[half_n - 1, 0].get_position().y0) / 2,
             class_labels[0], fontsize=8, ha='right', va='center', rotation=90)
    fig.text(x_label, (second_position.y1 + axes[-1, 0].get_position().y0) / 2,
             class_labels[1], fontsize=8, ha='right', va='center', rotation=90)
    y_divider = (axes[half_n - 1, 0].get_position().y0 + second_position.y1) / 2
    x0 = axes[0, 0].get_position().x0
    x1 = axes[0, -1].get_position().x1
    fig.add_artist(Line2D([x0, x1], [y_divider, y_divider],
                          color='black', linewidth=1.5, transform=fig.transFigure))
  else:
    first_position = axes[0, 0].get_position()
    second_position = axes[0, half_n].get_position()
    y_title = first_position.y1 + 0.02
    fig.text((first_position.x0 + axes[0, half_n - 1].get_position().x1) / 2,
             y_title, class_labels[0], fontsize=8, ha='center', va='bottom')
    fig.text((second_position.x0 + axes[0, -1].get_position().x1) / 2,
             y_title, class_labels[1], fontsize=8, ha='center', va='bottom')
    x_divider = (axes[0, half_n - 1].get_position().x1 + second_position.x0) / 2
    y0 = axes[-1, 0].get_position().y0
    y1 = axes[0, 0].get_position().y1
    fig.add_artist(Line2D([x_divider, x_divider], [y0, y1],
                          color='black', linewidth=1.5, transform=fig.transFigure))

  plt.savefig(out_file, dpi=300, bbox_inches='tight')
  plt.savefig(out_file.replace('.png', '.pdf'), bbox_inches='tight')
  plt.close(fig)


def extract_nw_prob(X_S, y_S, X_T, y_T, noise_type, noise_sd,
                    nw_bandwidth, autoencoder, out_file, alpha, gamma, nu,
                    nw_kernel='epanechnikov', seed=42, loss='cross_entropy'):
  """Compute bridge-weighted NW probabilities and save them as CSV."""
  Z_S, Z_T_noise = _standardize_source_and_targets(
    X_S, X_T, noise_type, noise_sd, autoencoder, seed
  )
  likelihood_ratio = estimate_likelihood_ratio(Z_S, Z_T_noise, Z_S)
  nw_train = WeightedNadarayaWatson(Z_S, bandwidth=nw_bandwidth,
                                    kernel=nw_kernel, clip=(0, 1))
  nw_test = WeightedNadarayaWatson(Z_S, X_test=Z_T_noise,
                                   bandwidth=nw_bandwidth, kernel=nw_kernel, clip=(0, 1))

  geometry_count = 2 + len(gamma)
  if len(nu) == 1:
    nu = np.repeat(nu, geometry_count)
  elif len(nu) != geometry_count:
    raise ValueError(f"Length of nu must be 1 or match the number of geometries ({geometry_count}).")

  with open(out_file, 'w', newline='') as csv_handle:
    writer = csv.writer(csv_handle)
    writer.writerow(['name', 'alpha', 'gamma', 'nu', 'values...'])

    def write_vector(name, values, alpha_value=np.nan, gamma_value=np.nan, nu_value=np.nan):
      writer.writerow([name, alpha_value, gamma_value, nu_value, *np.asarray(values, dtype=float)])

    for alpha_value in alpha:
      bridge_specs = [
        ('kl_kl', None, nu[0]),
        ('hd_hd', None, nu[1]),
        *[('kl_hd', gamma_value, nu[index + 2])
          for index, gamma_value in enumerate(gamma)],
      ]
      for geometry, gamma_value, nu_value in bridge_specs:
        bridge_weights = get_bridge_weights(
          likelihood_ratio, geometry=geometry, alpha=alpha_value, gamma=gamma_value
        )
        geometry_name = geometry if gamma_value is None else f'{geometry}_{gamma_value}'
        write_vector(f'wgt_agn_{geometry_name}', bridge_weights,
                     alpha_value, gamma_value)
        write_vector(f'prob_agn_{geometry_name}', nw_test.predict(y_S, bridge_weights),
                     alpha_value, gamma_value)

        probabilities, adversarial_weights = fit_loss_adversarial_nw(
          nw_train, nw_test, y_S, likelihood_ratio,
          geometry=geometry, loss=loss, alpha=alpha_value,
          gamma=gamma_value, nu=nu_value,
          max_iter=_ADV_MAX_ITER, iter_conv_tol=_ADV_ITER_CONV_TOL,
          step_size=_ADV_STEP_SIZE,
        )
        write_vector(f'wgt_adv_{geometry_name}', adversarial_weights,
                     alpha_value, gamma_value, nu_value)
        write_vector(f'prob_adv_{geometry_name}', probabilities,
                     alpha_value, gamma_value, nu_value)

  print(f"Probabilities, weights, and latent matrices saved to {out_file}")


def main():
  parser = argparse.ArgumentParser(description='Visualize Fashion-MNIST domain shifts')
  parser.add_argument('--class_A', '-a', default='Pullover')
  parser.add_argument('--class_B', '-b', default='Coat')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--alpha', type=parse_range, default=[0.5])
  parser.add_argument('--gamma', type=parse_range, default=[1.0])
  parser.add_argument('--nw_bw', type=float, default=1.0)
  parser.add_argument('--nu', type=parse_range, default=[1.0])
  parser.add_argument('--noise_type', choices=['laplace', 'gaussian'], default='laplace')
  parser.add_argument('--noise_sd', type=_positive_float, default=0.3)
  parser.add_argument('--num_images', type=int, default=6)
  parser.add_argument('--ae', help='Path to the trained autoencoder model file.')
  parser.add_argument('--ae_dim', type=int, default=10)
  parser.add_argument('--images-direction', choices=['h', 'v'], default='h')
  parser.add_argument('--out_path', default='figures')
  parser.add_argument('type', choices=['images', 'latent', 'compute_probabilities'])
  args = parser.parse_args()

  out_file = os.path.join(
    args.out_path,
    f'{args.class_A}_{args.class_B}_noise={args.noise_sd:.2f}_{args.type}.png'
  )
  os.makedirs(args.out_path, exist_ok=True)
  fmnist_data = load_fmnist(args.class_A, args.class_B)

  ae = None
  if args.ae is not None:
    ae = AutoencoderFMNIST(output_dim=args.ae_dim)
    ae.load_state_dict(torch.load(args.ae))
    ae.eval()

  if args.type == 'images':
    visualize_degradations(
      X_clean=fmnist_data['X_train'], y=fmnist_data['y_train'],
      out_file=out_file, noise_sd=args.noise_sd, noise_type=args.noise_type,
      class_labels=(args.class_A, args.class_B), num_images=args.num_images,
      seed=args.seed, direction=args.images_direction,
    )
  elif args.type == 'latent':
    if ae is None:
      parser.error('--ae is required for the latent visualization')
    plot_latent_shift_target(
      X_S=fmnist_data['X_train'], y_S=fmnist_data['y_train'],
      X_T=fmnist_data['X_test'], y_T=fmnist_data['y_test'],
      noise_sd=args.noise_sd, autoencoder=ae, out_file=out_file,
      seed=args.seed, noise_type=args.noise_type,
      class_labels=(args.class_A, args.class_B),
    )
  else:
    if ae is None:
      parser.error('--ae is required for compute_probabilities')
    out_file = os.path.join(
      args.out_path,
      f'{args.class_A}_{args.class_B}_noise={args.noise_type}_{args.noise_sd:.2f}_'
      f'{args.type}_al{"_".join(f"{value:.2f}" for value in args.alpha)}_'
      f'ga{"_".join(f"{value:.1f}" for value in args.gamma)}_'
      f'nu{"_".join(f"{value:.4f}" for value in args.nu)}_'
      f'bw{args.nw_bw:.2f}_seed{args.seed}.csv'
    )
    extract_nw_prob(
      X_S=fmnist_data['X_train'], y_S=fmnist_data['y_train'],
      X_T=fmnist_data['X_test'], y_T=fmnist_data['y_test'],
      noise_type=args.noise_type, noise_sd=args.noise_sd,
      autoencoder=ae, nw_bandwidth=args.nw_bw, out_file=out_file,
      alpha=args.alpha, gamma=args.gamma, nu=args.nu, seed=args.seed,
    )


if __name__ == '__main__':
  main()
