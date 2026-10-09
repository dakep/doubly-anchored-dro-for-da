import numpy as np
from scipy.stats import gaussian_kde, log
from scipy.special import wrightomega
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import torch

class EstimatedLikelihoodRatio:
  """Class for estimating likelihood ratios using Gaussian KDE."""

  def __init__(self, lr: np.ndarray, log: bool, method: str):
    self.lr = lr
    self.log = log
    self.method = method

  def val(self, log: bool = False) -> np.ndarray:
    """Returns the estimated likelihood ratio or its log."""
    if log:
      return np.log(self.lr) if not self.log else self.lr
    else:
      return self.lr if not self.log else np.exp(self.lr)

def estimate_likelihood_ratio(X_source: np.ndarray, X_target: np.ndarray, X_eval: np.ndarray, log: bool = True, bw_method: str|None='silverman'):
    """Estimates LR(x) = p_T(x) / p_S(x) or the log of that using Gaussian KDE."""
    # Transpose for scipy gaussian_kde (expects dims x samples)
    kde_S = gaussian_kde(X_source.T, bw_method=bw_method)
    kde_T = gaussian_kde(X_target.T, bw_method=bw_method)

    p_S = kde_S(X_eval.T)
    p_T = kde_T(X_eval.T)

    if log:
      lr = EstimatedLikelihoodRatio(np.log(p_T) - np.log(p_S), log=True,
                                    method=f'gaussian KDE with bandwidth {bw_method}')
    else:
      lr = EstimatedLikelihoodRatio(p_T / p_S, log=False,
                                    method=f'gaussian KDE with bandwidth {bw_method}')
    return lr

class WeightedNadarayaWatson:
  """Weighted NW regressor with cached RBF kernel matrices."""

  def __init__(self, X_train: np.ndarray, kernel: str = 'epanechnikov', bandwidth: float = 1.0,
               min_neighbors: int = 3, X_test: np.ndarray | None = None, clip: tuple[float, float] | None = None):
    X_eval = X_train if X_test is None else X_test
    self.clip = clip
    if kernel == "rbf":
      gamma_val = 1.0 / (2 * bandwidth ** 2)
      self.K_eval = rbf_kernel(X_eval, X_train, gamma=gamma_val)
    elif kernel in ("epanechnikov", "tricube"):
      pds = pairwise_distances(X_eval, X_train, metric="euclidean")

      # min_neighbors - 1 gives the index of the kth neighbor (0-indexed)
      k_index = min_neighbors - 1

      # np.partition is much faster than np.sort. It guarantees the element at k_index
      # is the same as if the array were sorted.
      kth_distances = np.partition(pds, k_index, axis=1)[:, k_index]

      # Expand the bandwidth for points where the kth neighbor is outside the default window.
      # We add 1e-8 because finite kernels evaluate to exactly 0 at u=1. To get a non-zero
      # weight for the 3rd neighbor, the bandwidth must be strictly larger than its distance.
      adaptive_bw = np.maximum(bandwidth, kth_distances + 1e-8)

      # Scale distances by each point's unique bandwidth using broadcasting
      u = pds / adaptive_bw[:, np.newaxis]
      if kernel == "epanechnikov":
          self.K_eval = (1.0 - u**2) # (3.0 / 4.0) # normalization not needed because it cancels out in NW prediction
      elif kernel == "tricube":
          self.K_eval = (1.0 - u**3)**3 # (70.0 / 81.0) # normalization not needed because it cancels out in NW prediction

      self.K_eval[u > 1.0] = 0.0
    else:
      raise ValueError(f"Unsupported kernel: {kernel}")

  def predict(
      self,
      y: np.ndarray,
      weights: np.ndarray | None = None,
      clip: tuple[float, float] | None = None
  ) -> np.ndarray:
    weighted_K = self.K_eval if weights is None else self.K_eval * weights
    numerator = np.dot(weighted_K, y)
    denominator = np.sum(weighted_K, axis=1) + 1e-8
    apply_clip = self.clip if clip is None else clip
    return np.clip(numerator / denominator, apply_clip[0], apply_clip[1]) if apply_clip is not None else \
      (numerator / denominator)

def get_loss_adversarial_weights(lr: EstimatedLikelihoodRatio, geometry: str, losses: np.ndarray|None=None,
                                 alpha: float|None=None, gamma: float=0.0, nu: float|None=None,
                                 num_tol: float=1e-8) -> tuple[np.ndarray, int]:
  """
  Computes the loss-adversarial weights dynamically incorporating empirical risk.
  """
  if alpha is None:
    raise ValueError("alpha must be provided.")

  ignore_losses = False
  if losses is None:
    ignore_losses = True
    losses = 0.0
    nu = 1.0  # Default nu when losses are ignored

  num_tol_log = np.log(num_tol)
  if num_tol_log > 0:
    raise ValueError("num_tol must be a small positive value less than 1.")

  if geometry == 'kl_kl':
    # Exponential tilting + geometric bridge
    log_lr = lr.val(log=True)
    lr_component = np.zeros_like(log_lr) if alpha == 0 else alpha * log_lr
    log_wgts = losses / nu + lr_component

  elif geometry == 'hd_hd':
    # Rational tilting + square-root stabilization
    if not ignore_losses:
      max_loss = np.max(losses)
      if nu is None:
        # Enforce the strict positivity constraint: c > sup L_i(h)
        nu = max_loss + np.std(losses)
      elif nu <= max_loss:
        print(f"WARNING: nu must be greater than the maximum loss ({max_loss:.4f}) for 'hd_hd' geometry. Adjusting nu to {max_loss + 1e-4:.4f}.")
        raise ValueError(f"nu must be greater than the maximum loss ({max_loss:.4f}) for 'hd_hd' geometry.")

    log_lr = lr.val(log=True)
    if alpha == 0:
      log_mixture = np.zeros_like(log_lr)
    elif alpha == 1:
      log_mixture = 0.5 * log_lr
    else:
      log_mixture = np.logaddexp(np.log1p(-alpha), np.log(alpha) + 0.5 * log_lr)
    log_wgts = 2 * (log_mixture - np.log(nu - losses))

  elif geometry == 'kl_hd':
    # Lambert-W tilting + asymmetric stabilization
    # beta = (alpha / (1 - alpha)) * np.sqrt(lr)
    # A_h = np.clip((losses + gamma) / nu, -20, 20)
    # arg = beta * np.exp(-A_h)
    beta_log = np.log(alpha) - np.log(1 - alpha) + 0.5 * lr.val(log=True)
    A_h_beta_log = beta_log - (losses + gamma) / nu
    with np.errstate(divide='ignore'):
      W_val = np.log(np.real(wrightomega(A_h_beta_log)))
    underflow = np.isneginf(W_val) & np.isfinite(A_h_beta_log)
    W_val[underflow] = A_h_beta_log[underflow]

    log_wgts = np.empty_like(beta_log)
    positive_infinite = np.isposinf(beta_log)
    negative_infinite_pair = np.isneginf(beta_log) & np.isneginf(W_val)
    log_wgts[positive_infinite] = np.inf
    log_wgts[negative_infinite_pair] = 0
    finite_pair = ~(positive_infinite | negative_infinite_pair)
    log_wgts[finite_pair] = 2 * (beta_log[finite_pair] - W_val[finite_pair])
  else:
    raise ValueError("Geometry must be 'kl_kl', 'hd_hd', or 'kl_hd'")

  nr_capped_weights = np.sum(log_wgts > -num_tol_log)
  log_wgts = np.clip(log_wgts, None, -num_tol_log)
  # Normalize weights to sum to N (maintains stability in the NW denominator)
  norm_log_wgts = log_wgts + np.log(len(log_wgts)) - np.log(np.sum(np.exp(log_wgts)))
  return np.exp(norm_log_wgts), nr_capped_weights

def get_bridge_weights(lr: EstimatedLikelihoodRatio, geometry: str='kl_kl', alpha: float|None=None,
                       gamma: float|None=None) -> tuple[np.ndarray, int]:
    return get_loss_adversarial_weights(lr, losses=None, geometry=geometry, alpha=alpha, gamma=gamma)

def fit_loss_adversarial_nw(nw_train: WeightedNadarayaWatson, nw_test: WeightedNadarayaWatson,
                            y_train: torch.Tensor, lr_train: EstimatedLikelihoodRatio,
                            loss: str='mse', geometry: str='kl_kl', nu: float|None=None,
                            max_iter: int=5, iter_conv_tol: float=1e-4, step_size: float=0.5, min_step_size: float=1e-6,
                            **kwargs) -> tuple[np.ndarray, np.ndarray]:
  """
  Fits the NW estimator using an iteratively reweighted scheme to solve the
  profiled min-max optimization problem.
  """
  if loss == 'mse':
    loss_fn = lambda y_true, y_pred: (y_true - y_pred) ** 2
  elif loss == 'mae':
    loss_fn = lambda y_true, y_pred: np.abs(y_true - y_pred)
  elif loss == 'cross_entropy':
    loss_fn = lambda y_true, y_pred: np.clip(-(y_true * np.log(y_pred + 1e-8) +
                                               (1 - y_true) * np.log(1 - y_pred + 1e-8)),
                                             0, 4) # Clip to have bounded loss
  else:
    raise ValueError("Loss must be 'mse', 'mae', or 'cross_entropy'.")

  # 1. Initialize with pure density-bridge weights (structural baseline, no loss tilt)
  w_current, n_capped = get_bridge_weights(lr_train, geometry=geometry, **kwargs)
  w_current = w_current * (len(w_current) / np.sum(w_current))
  w_prev_2step = w_current.copy()  # For cycle detection

  iter_conv_tol *= step_size  # Adjust convergence tolerance based on step size

  # 2. Iteratively refine weights and predictions
  for iteration in range(max_iter):
    # Step A: Fit current predictor on source data to estimate local conditional risk
    y_pred_train = nw_train.predict(y_train, w_current)

    # Step B: Calculate empirical source loss
    losses = loss_fn(y_train, y_pred_train)
    conditional_risk = nw_train.predict(losses)

    # Step C: Update adversarial weights incorporating the new loss landscape
    try:
      w_adv, n_capped = get_loss_adversarial_weights(lr_train, losses=conditional_risk, geometry=geometry, nu=nu, **kwargs)
    except ValueError as e:
      print(f"Error in calculating adversarial weights: {e}")
      raise e

    w_new = (1 - step_size) * w_current + step_size * w_adv
    wdiff = np.linalg.norm(w_new - w_current)
    wdiff_2step = np.linalg.norm(w_new - w_prev_2step)  # For cycle detection

    # Step D: Check for convergence or cycle detection
    if wdiff < iter_conv_tol:
      print(f"Iteration {iteration + 1}/{max_iter} | Weight L2 Diff: {wdiff:.6f}, {wdiff_2step:.6f} | Converged!")
      break
    elif wdiff_2step < iter_conv_tol:
      if step_size < min_step_size:
        print(f"Iteration {iteration + 1}/{max_iter} | Cycle detected, but step size too small to reduce further. Stopping.")
        break

      step_size *= 0.5  # Reduce step size to escape potential cycle
      iter_conv_tol *= 0.5  # Adjust convergence tolerance accordingly
      print(f"Iteration {iteration + 1}/{max_iter} | Cycle detected, reducing step size to {step_size:.6f}!")
    else:
      print(f"Iteration {iteration + 1}/{max_iter} | Weight L2 Diff: {wdiff:.6f}, {wdiff_2step:.6f}")

    w_prev_2step = w_current.copy()
    w_current = w_new

  # 3. New prediction on the target domain using the updated adversarial weights
  y_pred_target = nw_test.predict(y_train, w_current)

  return y_pred_target, w_current, n_capped
