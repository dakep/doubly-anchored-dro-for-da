"""Run the bounded-overlap, theorem-linked domain-adaptation simulation.

By default this runs the fixed design from simulation-design.tex: n=m=2000,
replications 1 through 1000, and the 51-point alpha grid. Outputs are written to
results/simulation-design. Use --replications for a shorter validation run.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit

EPSILON = 0.5
LABEL_SHIFT = 0.5
SAMPLE_SIZE = 2000
TARGET_SAMPLE_SIZE = 2000
N_REPLICATIONS = 1000
N_THETA = 201
N_ALPHA = 51
DELTA = 0.05
GEOMETRIES = ("kl_kl", "hd_hd")
GEOMETRY_LABELS = {"kl_kl": "KL/KL", "hd_hd": "HD/HD"}


@dataclass(frozen=True)
class SimulationConfig:
  epsilon: float
  label_shift: float
  delta: float
  sample_size: int
  target_sample_size: int


def source_probability(covariates: np.ndarray | float, return_score: bool=False) -> np.ndarray | float:
  score = 3.0 * ((2.0 * covariates - 1.0) ** 2 - 0.5)
  return expit(score) if not return_score else score


def target_probability(
  covariates: np.ndarray | float,
  config: SimulationConfig,
) -> np.ndarray | float:
  score = source_probability(covariates, return_score=True) + config.label_shift * (2.0 * covariates - 1.0)
  return expit(score)


def density_ratio(
  covariates: np.ndarray | float,
  config: SimulationConfig,
) -> np.ndarray | float:
  return 1.0 + config.epsilon * (2.0 * covariates - 1.0)


def sample_target_covariates(
  uniform_draws: np.ndarray,
  config: SimulationConfig,
) -> np.ndarray:
  return (
    np.sqrt((1.0 - config.epsilon) ** 2 + 4.0 * config.epsilon * uniform_draws)
    - (1.0 - config.epsilon)
  ) / (2.0 * config.epsilon)


def bernoulli_kl(p: float, q: float) -> float:
  return p * (np.log(p) - np.log(q)) + (1.0 - p) * (np.log(1.0 - p) - np.log(1.0 - q))

def bernoulli_hellinger_squared(p: float, q: float) -> float:
  return (np.sqrt(p) - np.sqrt(q)) ** 2 + (np.sqrt(1.0 - p) - np.sqrt(1.0 - q)) ** 2


def kl_log_moment(
  alpha_values: np.ndarray | float,
  config: SimulationConfig,
) -> np.ndarray | float:
  lower = 1.0 - config.epsilon
  upper = 1.0 + config.epsilon
  power = np.asarray(alpha_values) + 1.0
  moment = (upper**power - lower**power) / (2.0 * config.epsilon * power)
  return np.log(moment)


def radius_profiles(
  geometry: str,
  alpha_values: np.ndarray | float,
  config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
  alpha_values = np.asarray(alpha_values, dtype=float)
  lower = 1.0 - config.epsilon
  upper = 1.0 + config.epsilon

  if geometry == "kl_kl":
    power = alpha_values + 1.0
    log_moment = kl_log_moment(alpha_values, config)
    derivative = (
      (upper**power * np.log(upper) - lower**power * np.log(lower))
      / (upper**power - lower**power)
      - 1.0 / power
    )
    source_radius = alpha_values * derivative - log_moment
    target_radius = (alpha_values - 1.0) * derivative - log_moment
    return np.maximum(source_radius, 0.0), np.maximum(target_radius, 0.0)

  if geometry == "hd_hd":
    hellinger_covariate = 2.0 - 2.0 * (upper**1.5 - lower**1.5) / (3.0 * config.epsilon)
    normalizer = 1.0 - alpha_values * (1.0 - alpha_values) * hellinger_covariate
    source_radius = 2.0 - (2.0 - alpha_values * hellinger_covariate) / np.sqrt(normalizer)
    target_radius = 2.0 - (2.0 - (1.0 - alpha_values) * hellinger_covariate) / np.sqrt(normalizer)
    return np.maximum(source_radius, 0.0), np.maximum(target_radius, 0.0)

  raise ValueError(f"Unsupported geometry: {geometry}")


def quadrature_radius_profiles(
  geometry: str,
  alpha_value: float,
  config: SimulationConfig,
) -> tuple[float, float]:
  lower = 1.0 - config.epsilon
  upper = 1.0 + config.epsilon

  if geometry == "kl_kl":
    normalizer = np.exp(kl_log_moment(alpha_value, config))

    def bridge_density(covariate: float) -> float:
      return density_ratio(covariate, config) ** alpha_value / normalizer

    source_radius = quad(
      lambda covariate: bridge_density(covariate) * np.log(bridge_density(covariate)),
      0.0,
      1.0,
      epsabs=1e-13,
      epsrel=1e-13,
    )[0]
    target_radius = quad(
      lambda covariate: bridge_density(covariate)
      * np.log(bridge_density(covariate) / density_ratio(covariate, config)),
      0.0,
      1.0,
      epsabs=1e-13,
      epsrel=1e-13,
    )[0]
    return source_radius, target_radius

  if geometry == "hd_hd":
    hellinger_covariate = 2.0 - 2.0 * (upper**1.5 - lower**1.5) / (3.0 * config.epsilon)
    normalizer = 1.0 - alpha_value * (1.0 - alpha_value) * hellinger_covariate

    def bridge_density(covariate: float) -> float:
      bridge_amplitude = (
        1.0 - alpha_value
        + alpha_value * np.sqrt(density_ratio(covariate, config))
      )
      return bridge_amplitude**2 / normalizer

    source_affinity = quad(
      lambda covariate: np.sqrt(bridge_density(covariate)),
      0.0,
      1.0,
      epsabs=1e-13,
      epsrel=1e-13,
    )[0]
    target_affinity = quad(
      lambda covariate: np.sqrt(
        density_ratio(covariate, config) * bridge_density(covariate)
      ),
      0.0,
      1.0,
      epsabs=1e-13,
      epsrel=1e-13,
    )[0]
    return 2.0 - 2.0 * source_affinity, 2.0 - 2.0 * target_affinity

  raise ValueError(f"Unsupported geometry: {geometry}")


def target_label_divergences(config: SimulationConfig) -> tuple[float, float]:
  def kl_integrand(covariate: float) -> float:
    return density_ratio(covariate, config) * bernoulli_kl(
      target_probability(covariate, config), source_probability(covariate)
    )

  def hellinger_integrand(covariate: float) -> float:
    return density_ratio(covariate, config) * bernoulli_hellinger_squared(
      target_probability(covariate, config), source_probability(covariate)
    )

  target_kl = quad(kl_integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)[0]
  target_hellinger = quad(hellinger_integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)[0]
  return target_kl, target_hellinger


def population_divergences(
  label_kl: float,
  label_hellinger: float,
  config: SimulationConfig,
) -> dict[str, float]:
  kl_total = quad(
    lambda covariate: density_ratio(covariate, config)
    * (
      np.log(density_ratio(covariate, config))
      + bernoulli_kl(
        target_probability(covariate, config), source_probability(covariate)
      )
    ),
    0.0,
    1.0,
    epsabs=1e-13,
    epsrel=1e-13,
  )[0]
  hellinger_affinity = quad(
    lambda covariate: np.sqrt(density_ratio(covariate, config))
    * (
      np.sqrt(
        target_probability(covariate, config) * source_probability(covariate)
      )
      + np.sqrt(
        (1.0 - target_probability(covariate, config))
        * (1.0 - source_probability(covariate))
      )
    ),
    0.0,
    1.0,
    epsabs=1e-13,
    epsrel=1e-13,
  )[0]
  hellinger_total = 2.0 - 2.0 * hellinger_affinity
  return {
    "kl_kl": kl_total,
    "hd_hd": hellinger_total,
    "label_kl": label_kl,
    "label_hellinger": label_hellinger,
  }


def exact_target_risks(
  theta_values: np.ndarray,
  config: SimulationConfig,
) -> np.ndarray:
  risks = np.empty(len(theta_values), dtype=float)
  for theta_index, theta_value in enumerate(theta_values):
    def risk_integrand(covariate: float) -> float:
      target_label_probability = target_probability(covariate, config)
      prediction = 0.5 + theta_value * (2.0 * covariate - 1.0)
      conditional_risk = (
        target_label_probability * (1.0 - prediction) ** 2
        + (1.0 - target_label_probability) * prediction**2
      )
      return density_ratio(covariate, config) * conditional_risk

    risks[theta_index] = quad(
      risk_integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13
    )[0]
  return risks


def structural_weights(geometry: str, ratios: np.ndarray, alpha_value: float) -> np.ndarray:
  if geometry == "kl_kl":
    return ratios**alpha_value
  if geometry == "hd_hd":
    return (1.0 - alpha_value + alpha_value * np.sqrt(ratios)) ** 2
  raise ValueError(f"Unsupported geometry: {geometry}")


def empirical_erm_index(
  covariates: np.ndarray,
  labels: np.ndarray,
  theta_values: np.ndarray,
  weights: np.ndarray,
) -> int:
  centered_labels = labels - 0.5
  feature = 2.0 * covariates - 1.0
  sum_weights = np.sum(weights)
  risk_constant = np.dot(weights, centered_labels**2)
  risk_linear = np.dot(weights, feature * centered_labels)
  risk_quadratic = np.dot(weights, feature**2)
  empirical_risks = (
    risk_constant
    - 2.0 * risk_linear * theta_values
    + risk_quadratic * theta_values**2
  ) / sum_weights
  return int(np.argmin(empirical_risks))

def check_implementation(
  alpha_grid: np.ndarray,
  theta_values: np.ndarray,
  population: dict[str, dict[str, float | np.ndarray]],
  config: SimulationConfig,
) -> None:
  profile_errors = []
  endpoint_errors = []
  monotonicity_ok = True

  for geometry in GEOMETRIES:
    source_profile, target_profile = radius_profiles(geometry, alpha_grid, config)
    for alpha_index, alpha_value in enumerate(alpha_grid):
      direct_source, direct_target = quadrature_radius_profiles(geometry, alpha_value, config)
      profile_errors.extend((
        abs(source_profile[alpha_index] - direct_source),
        abs(target_profile[alpha_index] - direct_target),
      ))
    endpoint_errors.extend((abs(source_profile[0]), abs(target_profile[-1])))
    monotonicity_ok &= bool(
      np.all(np.diff(source_profile) >= -1e-12)
      and np.all(np.diff(target_profile) <= 1e-12)
    )

  max_profile_error = max(profile_errors)
  print(
    f"1. Closed-form profiles agree with direct quadrature within 1e-10: "
    f"{'PASS' if max_profile_error <= 1e-10 else 'FAIL'} "
    f"(max absolute error {max_profile_error:.3e})."
  )
  endpoints_ok = max(endpoint_errors) <= 1e-12
  print(
    f"2. Radius endpoints and monotonicities: "
    f"{'PASS' if endpoints_ok and monotonicity_ok else 'FAIL'} "
    f"(max endpoint error {max(endpoint_errors):.3e}; monotone={monotonicity_ok})."
  )

  root_residuals = []
  for geometry in GEOMETRIES:
    alpha_lower = population[geometry]["alpha_lower"]
    target_at_root = radius_profiles(geometry, alpha_lower, config)[1]
    root_residuals.append(abs(float(target_at_root) - population[geometry]["rho_target"]))
  max_root_residual = max(root_residuals)
  print(
    f"3. Lower-endpoint root residual is at most 1e-10: "
    f"{'PASS' if max_root_residual <= 1e-10 else 'FAIL'} "
    f"(max absolute residual {max_root_residual:.3e})."
  )

  check_rng = np.random.default_rng(1)
  check_covariates = check_rng.uniform(0.0, 1.0, config.sample_size)
  check_labels = check_rng.binomial(1, source_probability(check_covariates))
  unweighted_index = empirical_erm_index(
    check_covariates, check_labels, theta_values, np.ones(config.sample_size)
  )
  alpha_zero_ok = all(
    empirical_erm_index(
      check_covariates,
      check_labels,
      theta_values,
      structural_weights(
        geometry, density_ratio(check_covariates, config), 0.0
      ),
    ) == unweighted_index
    for geometry in GEOMETRIES
  )
  print(
    f"4. Alpha=0 reproduces the unweighted source ERM: "
    f"{'PASS' if alpha_zero_ok else 'FAIL'}."
  )

  check_ratios = density_ratio(np.linspace(0.0, 1.0, 201), config)
  kl_importance = structural_weights("kl_kl", check_ratios, 1.0)
  hd_importance = structural_weights("hd_hd", check_ratios, 1.0)
  kl_importance /= np.sum(kl_importance)
  hd_importance /= np.sum(hd_importance)
  importance_match = np.allclose(kl_importance, hd_importance, rtol=1e-13, atol=1e-15)
  print(
    f"5. Alpha=1 normalized KL/KL and HD/HD importance weights agree: "
    f"{'PASS' if importance_match else 'FAIL'} "
    f"(max absolute difference {np.max(np.abs(kl_importance - hd_importance)):.3e})."
  )

def build_population_summary(
  alpha_grid: np.ndarray,
  config: SimulationConfig,
) -> dict[str, dict[str, float | np.ndarray]]:
  label_radii = target_label_divergences(config)
  divergences = population_divergences(label_radii[0], label_radii[1], config)
  theta_values = np.linspace(-0.5, 0.5, N_THETA)
  risks_by_theta = exact_target_risks(theta_values, config)
  reference_risk = float(np.min(risks_by_theta))

  kappa = np.log(80.0 * N_ALPHA * N_THETA / config.delta)
  concentration_radius = (
    2.0 * np.sqrt(18.0 * kappa / (config.sample_size - 1.0))
    + 30.0 * kappa / (config.sample_size - 1.0)
  )
  lower_ratio = 1.0 - config.epsilon
  upper_ratio = 1.0 + config.epsilon
  ratio_range = upper_ratio / lower_ratio
  hellinger_covariate = 2.0 - 2.0 * (
    upper_ratio**1.5 - lower_ratio**1.5
  ) / (3.0 * config.epsilon)

  population = {}
  for geom_i, geometry in enumerate(GEOMETRIES):
    divergence = divergences[geometry]
    source_profile, target_profile = radius_profiles(geometry, alpha_grid, config)
    source_radius = 1.10 * divergence
    label_radius = label_radii[geom_i]
    # target_radius = 0.5 * float(radius_profiles(geometry, 0.0, config)[1])
    target_radius = label_radius + 0.25 * (
      float(radius_profiles(geometry, 0.0, config)[1]) - label_radius
    )
    alpha_lower = brentq(
      lambda alpha_value: float(radius_profiles(geometry, alpha_value, config)[1]) - target_radius,
      0.0,
      1.0,
    )
    alpha_upper = (
      brentq(
        lambda alpha_value: float(radius_profiles(geometry, alpha_value, config)[0]) - source_radius,
        0.0,
        1.0,
      )
      if float(source_profile[-1]) > source_radius
      else 1.0
    )

    if geometry == "kl_kl":
      log_ratio_mean = quad(
        lambda covariate: density_ratio(covariate, config) * np.log(density_ratio(covariate, config)),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
      )[0]
      log_ratio_second = quad(
        lambda covariate: density_ratio(covariate, config) * np.log(density_ratio(covariate, config)) ** 2,
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
      )[0]
      log_ratio_variance = max(0.0, log_ratio_second - log_ratio_mean**2)
      endpoint_margin = (
        concentration_radius * ratio_range * np.log(ratio_range)
        - 0.5 * np.sqrt(log_ratio_variance)
      )
      envelope_radius = np.sqrt(target_profile / 2.0)
      population_ess = np.exp(
        2.0 * kl_log_moment(alpha_grid, config)
        - kl_log_moment(2.0 * alpha_grid, config)
      )
      sensitivity = ratio_range**alpha_grid
    else:
      endpoint_margin = (
        concentration_radius
        * (2.0 * np.sqrt(ratio_range) * (np.sqrt(upper_ratio) - np.sqrt(lower_ratio)) / lower_ratio)
        - np.sqrt(hellinger_covariate * (1.0 - hellinger_covariate / 4.0))
      )
      envelope_radius = np.sqrt(target_profile)
      normalizer = 1.0 - alpha_grid * (1.0 - alpha_grid) * hellinger_covariate
      second_moment = np.empty_like(alpha_grid)
      for alpha_index, alpha_value in enumerate(alpha_grid):
        second_moment[alpha_index] = quad(
          lambda covariate: (
            1.0 - alpha_value
            + alpha_value * np.sqrt(density_ratio(covariate, config))
          ) ** 4,
          0.0,
          1.0,
          epsabs=1e-13,
          epsrel=1e-13,
        )[0]
      population_ess = normalizer**2 / second_moment
      sensitivity = (
        (1.0 - alpha_grid + alpha_grid * np.sqrt(upper_ratio))
        / (1.0 - alpha_grid + alpha_grid * np.sqrt(lower_ratio))
      ) ** 2

    envelope = (
      2.0 * envelope_radius
      + 2.0 * np.sqrt(label_radius / (2.0 if geometry == "kl_kl" else 1.0))
      + 2.0 * sensitivity * concentration_radius
    )
    population[geometry] = {
      "divergence": divergence,
      "label_radius": label_radius,
      "rho_source": source_radius,
      "rho_target": target_radius,
      "alpha_lower": alpha_lower,
      "alpha_upper": alpha_upper,
      "endpoint_margin": endpoint_margin,
      "source_profile": source_profile,
      "target_profile": target_profile,
      "envelope": envelope,
      "population_ess": population_ess,
      "reference_risk": reference_risk,
      "target_risks_by_theta": risks_by_theta,
      "theta_values": theta_values,
      "concentration_radius": concentration_radius,
    }
  return population


def run_simulation(
  replications: int,
  start_replication: int,
  alpha_grid: np.ndarray,
  population: dict[str, dict[str, float | np.ndarray]],
  config: SimulationConfig,
) -> pd.DataFrame:
  rows = []
  theta_values = population["kl_kl"]["theta_values"]
  reference_risk = population["kl_kl"]["reference_risk"]
  target_risks_by_theta = population["kl_kl"]["target_risks_by_theta"]

  for replication in range(start_replication, start_replication + replications):
    print(f"Running replication {replication}...")
    rng = np.random.default_rng(seed=replication)
    source_covariates = rng.uniform(0.0, 1.0, config.sample_size)
    source_labels = rng.binomial(1, source_probability(source_covariates))
    target_covariates = sample_target_covariates(
      rng.uniform(0.0, 1.0, config.target_sample_size), config
    )
    source_ratios = density_ratio(source_covariates, config)
    feature = 2.0 * source_covariates - 1.0
    centered_labels = source_labels - 0.5
    constant_loss = centered_labels**2

    for geometry in GEOMETRIES:
      weight_matrix = structural_weights(
        geometry, source_ratios[None, :], alpha_grid[:, None]
      )
      sum_weights = np.sum(weight_matrix, axis=1)
      risk_constant = weight_matrix @ constant_loss
      risk_linear = weight_matrix @ (feature * centered_labels)
      risk_quadratic = weight_matrix @ (feature**2)
      empirical_risks = (
        risk_constant[:, None]
        - 2.0 * risk_linear[:, None] * theta_values[None, :]
        + risk_quadratic[:, None] * theta_values[None, :] ** 2
      ) / sum_weights[:, None]
      best_indices = np.argmin(empirical_risks, axis=1)
      chosen_theta = theta_values[best_indices]
      target_excess = target_risks_by_theta[best_indices] - reference_risk
      effective_sample_size_fraction = (
        sum_weights**2 / np.sum(weight_matrix**2, axis=1) / config.sample_size
      )

      for alpha_index, alpha_value in enumerate(alpha_grid):
        rows.append({
          "replication": replication,
          "seed": replication,
          "n_source": config.sample_size,
          "n_target": config.target_sample_size,
          "epsilon": config.epsilon,
          "label_shift": config.label_shift,
          "delta": config.delta,
          "geometry": geometry,
          "alpha": alpha_value,
          "theta_hat": chosen_theta[alpha_index],
          "target_excess_risk": target_excess[alpha_index],
          "ess_fraction": effective_sample_size_fraction[alpha_index],
          "source_x_mean": np.mean(source_covariates),
          "source_y_mean": np.mean(source_labels),
          "target_x_mean": np.mean(target_covariates),
        })

    if (replication - start_replication + 1) % 100 == 0:
      print(f"Completed replication {replication}.")

  return pd.DataFrame(rows)


def make_summary(
  results: pd.DataFrame,
  alpha_grid: np.ndarray,
  population: dict[str, dict[str, float | np.ndarray]],
  config: SimulationConfig,
) -> pd.DataFrame:
  risk_summary = results.groupby(["geometry", "alpha"])["target_excess_risk"].agg(
    mean="mean", q10=lambda values: values.quantile(0.10), q90=lambda values: values.quantile(0.90)
  ).reset_index()
  summary_rows = []

  for geometry in GEOMETRIES:
    geometry_data = population[geometry]
    risk_curve = risk_summary[risk_summary["geometry"] == geometry].sort_values("alpha")
    envelope = geometry_data["envelope"]
    assert isinstance(envelope, np.ndarray)
    summary_rows.append({
      "geometry": geometry,
      "epsilon": config.epsilon,
      "label_shift": config.label_shift,
      "delta": config.delta,
      "sample_size": config.sample_size,
      "target_sample_size": config.target_sample_size,
      "delta_target_label": geometry_data["label_radius"],
      "rho_source": geometry_data["rho_source"],
      "rho_target": geometry_data["rho_target"],
      "alpha_lower": geometry_data["alpha_lower"],
      "alpha_upper": geometry_data["alpha_upper"],
      "alpha_min_mean_excess_risk": float(risk_curve.iloc[np.argmin(risk_curve["mean"].to_numpy())]["alpha"]),
      "alpha_min_risk_envelope": float(alpha_grid[np.argmin(envelope)]),
      "endpoint_slope_margin": geometry_data["endpoint_margin"],
    })

  return pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--replications", type=int, default=N_REPLICATIONS)
  parser.add_argument("--start-replication", type=int, default=1)
  parser.add_argument("--output-dir", type=Path, default=Path("sim-results"))
  parser.add_argument("--check", action="store_true", help="Run implementation checks and exit")
  parser.add_argument("--epsilon", type=float, default=EPSILON)
  parser.add_argument("--label-shift", type=float, default=LABEL_SHIFT)
  parser.add_argument("--delta", type=float, default=DELTA)
  parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
  parser.add_argument("--target-sample-size", type=int, default=TARGET_SAMPLE_SIZE)
  args = parser.parse_args()
  if args.replications <= 0:
    parser.error("--replications must be positive")
  if args.start_replication <= 0:
    parser.error("--start-replication must be positive")
  if not 0.0 < args.epsilon < 1.0:
    parser.error("--epsilon must be between 0 and 1")
  if not np.isfinite(args.label_shift):
    parser.error("--label-shift must be finite")
  if not 0.0 < args.delta < 1.0:
    parser.error("--delta must be between 0 and 1")
  if args.sample_size <= 1:
    parser.error("--sample-size must be greater than 1")
  if args.target_sample_size <= 0:
    parser.error("--target-sample-size must be positive")
  return args


def main() -> None:
  args = parse_args()
  config = SimulationConfig(
    epsilon=args.epsilon,
    label_shift=args.label_shift,
    delta=args.delta,
    sample_size=args.sample_size,
    target_sample_size=args.target_sample_size,
  )
  alpha_grid = np.linspace(0, 1, num=N_ALPHA, dtype=np.float64)
  population = build_population_summary(alpha_grid, config)
  if args.check:
    check_implementation(alpha_grid, population["kl_kl"]["theta_values"], population, config)
    return

  results = run_simulation(
    args.replications, args.start_replication, alpha_grid, population, config
  )
  summary = make_summary(results, alpha_grid, population, config)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  results.to_csv(args.output_dir / f"simulation_replications_eps{args.epsilon:.2f}_c{args.label_shift:.2f}.csv", index=False)
  summary.to_csv(args.output_dir / f"simulation_summary_eps{args.epsilon:.2f}_c{args.label_shift:.2f}.csv", index=False)
  print(f"Simulation outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
  main()
