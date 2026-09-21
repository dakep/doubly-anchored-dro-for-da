import numpy as np
import torch

_SQRT_2 = 1.4142135623730951

def apply_gaussian_noise(X: torch.Tensor, noise_sd: float, rng : np.random.Generator = None):
  seed = int(rng.integers(0, 2**64, dtype=np.uint64)) if rng is not None else None
  generator = torch.Generator(device=X.device).manual_seed(seed)
  X_noisy = X + noise_sd * torch.randn(X.shape, generator=generator, device=X.device, dtype=X.dtype)
  return torch.clamp(X_noisy, 0., 1.)

def apply_laplace_noise(X: torch.Tensor, noise_sd: float, rng : np.random.Generator = None):
  seed = int(rng.integers(0, 2**64, dtype=np.uint64)) if rng is not None else None
  generator = torch.Generator(device=X.device).manual_seed(seed)
  unif_rand = torch.rand(X.shape, generator=generator, dtype=X.dtype, device=X.device) - 0.5
  laplace_noise = (noise_sd / _SQRT_2) * torch.sign(unif_rand) * torch.log(1.0 - 2.0 * torch.abs(unif_rand))
  X_noisy = X + laplace_noise
  return torch.clamp(X_noisy, 0., 1.)