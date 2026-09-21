import argparse

import os
import sys
import multiprocessing as mp
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

from da_image_degeneration import apply_gaussian_noise, apply_laplace_noise
from fashion_mnist_helper import load_fmnist, AutoencoderFMNIST
from bridged_nw import *

# adapt's KLIEP/ULSIF pull in TensorFlow, whose OpenMP runtime can segfault when
# loaded in the same process as PyTorch. Keep that import out of the main process
# and run those estimators in an isolated subprocess instead (see _run_reweighting_subprocess).
_TF_SUBPROCESS_ENV = {
  "KMP_DUPLICATE_LIB_OK": "TRUE",
  "TF_CPP_MIN_LOG_LEVEL": "3",
  "TF_NUM_INTRAOP_THREADS": "1",
  "TF_NUM_INTEROP_THREADS": "1",
  "OMP_NUM_THREADS": "1",
}

def _kliep_worker(Z_S, y_S, Z_T, seed, conn):
  os.environ.update(_TF_SUBPROCESS_ENV)
  try:
    from sklearn.dummy import DummyClassifier
    from adapt.instance_based import KLIEP
    kliep = KLIEP(estimator=DummyClassifier(), Xt=Z_T, random_state=seed)
    kliep.fit(Z_S, y_S)
    w = kliep.predict_weights(Z_S)
    conn.send(w * (len(w) / np.sum(w)))
  except Exception as e:
    conn.send(e)
  finally:
    conn.close()

def _ulsif_worker(Z_S, y_S, Z_T, seed, conn):
  os.environ.update(_TF_SUBPROCESS_ENV)
  try:
    from sklearn.dummy import DummyClassifier
    from adapt.instance_based import ULSIF
    ulsif = ULSIF(estimator=DummyClassifier(), Xt=Z_T, random_state=seed)
    ulsif.fit(Z_S, y_S)
    w = ulsif.predict_weights(Z_S)
    conn.send(w * (len(w) / np.sum(w)))
  except Exception as e:
    conn.send(e)
  finally:
    conn.close()

def _run_reweighting_subprocess(worker_fn, Z_S, y_S, Z_T, seed):
  ctx = mp.get_context("spawn")
  parent_conn, child_conn = ctx.Pipe()
  proc = ctx.Process(target=worker_fn, args=(Z_S, y_S, Z_T, seed, child_conn))
  proc.start()
  child_conn.close()
  try:
    result = parent_conn.recv()
  except EOFError:
    result = None
  proc.join()
  parent_conn.close()
  if proc.exitcode != 0:
    raise RuntimeError(f"{worker_fn.__name__} subprocess crashed (exit code {proc.exitcode})")
  if isinstance(result, Exception):
    raise result
  return result

# Set stdout and stderr to line-buffered mode for real-time output on HPCs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_ADV_MAX_ITER = 200
_ADV_STEP_SIZE = 0.2
_ADV_ITER_CONV_TOL = 1.0

def _positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("argument must be > 0")
  return parsed

def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("argument must be a positive integer")
  return parsed

def evaluate(name, y_true, y_prob, alpha: float=float('nan'), gamma: float=float('nan'), nu: float=float('nan')):
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= 0.5).astype(int))
    return {"model": name, "accuracy": acc, "auc": auc, "brier": brier, "f1": f1,
            "alpha": alpha, "gamma": gamma, "nu": nu}
    # print(f"{name:<25} | Accuracy: {acc:.4f} | Brier Score: {brier:.4f}")

def append_result_row(out_fname: str, row: dict) -> None:
  pd.DataFrame([row]).to_csv(
    out_fname,
    mode="a",
    header=not os.path.isfile(out_fname),
    index=False,
  )

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--class_A", "-a", type=str, default="Pullover")
  parser.add_argument("--class_B", "-b", type=str, default="Coat")
  parser.add_argument("--kde_bw", "-k", type=str, default='silverman')
  parser.add_argument("--nw_bw", "-n", type=float, default=2.0)
  parser.add_argument("--nw_kernel", "-nk", type=str, choices=['rbf', 'epanechnikov', 'tricube'], default='epanechnikov')
  parser.add_argument("--replications", "-r", type=_positive_int, default=100)
  parser.add_argument("--start_replication", "-s", type=_positive_int, default=1)
  parser.add_argument("--force_overwrite", '-f', action="store_true")
  parser.add_argument("--output_suffix", type=str, default="")
  parser.add_argument("--output_path", type=str, default="results")
  parser.add_argument("autoencoder", type=str)
  parser.add_argument("ae_dim", type=_positive_int)
  parser.add_argument("noise_type", choices=["laplace", "gaussian"])
  parser.add_argument("noise_sd", type=_positive_float)
  args = parser.parse_args()

  # Load the trained autoencoder
  ae = AutoencoderFMNIST(output_dim=args.ae_dim)
  ae.load_state_dict(torch.load(args.autoencoder))
  ae.eval()

  # Load the Fashion MNIST data for the specified classes
  fmnist_data = load_fmnist(args.class_A, args.class_B)

  out_fname = f"{args.output_path}/results_{args.class_A}_{args.class_B}_aedim{args.ae_dim}_{args.noise_type}_sd{args.noise_sd}_bw{args.nw_bw:0.2f}_kern{args.nw_kernel}_{args.output_suffix}.csv"

  if os.path.isfile(out_fname):
    if args.force_overwrite:
      os.remove(out_fname)
    else:
      raise ValueError(f"Output CSV {out_fname} already exists. Set --force_overwrite/-f to overwrite.")

  X_S = fmnist_data['X_train']
  y_S = fmnist_data['y_train']
  X_T_clean = fmnist_data['X_test']
  y_T = fmnist_data['y_test']

  def record_result(row: dict) -> None:
    append_result_row(out_fname, row)
    print(
      f"Accuracy={row['accuracy']:.4f} | "
      f"Brier score={row['brier']:.4f} | "
      f"AUC-ROC={row['auc']:.4f} | "
      f"F1 Score={row['f1']:.4f}"
    )

  with torch.no_grad():
    # Project into low dimensional Manifold
    Z_S = ae.encoder(X_S).numpy()

  scaler = StandardScaler().fit(Z_S)

  for r in range(args.start_replication, args.start_replication + args.replications):
    rng = np.random.default_rng(seed=r)  # Use a different seed for each replication

    print(f"Applying Domain Shift: {args.noise_type.capitalize()} Noise (SD = {args.noise_sd})")
    noise_fn = apply_gaussian_noise if args.noise_type == 'gaussian' else apply_laplace_noise

    with torch.no_grad():
      X_T_noisy = noise_fn(X_T_clean, noise_sd=args.noise_sd, rng=rng)
      # Project into low dimensional Manifold
      Z_T = ae.encoder(X_T_noisy).numpy()
      Z_T_clean = ae.encoder(X_T_clean).numpy()

    Z_S = scaler.transform(Z_S)
    Z_T = scaler.transform(Z_T)
    Z_T_clean = scaler.transform(Z_T_clean)

    try:
      kde_bw = float(args.kde_bw)
    except ValueError:
      kde_bw = args.kde_bw  # Use the string value directly (e.g., 'silverman')

    lr_kde = estimate_likelihood_ratio(Z_S, Z_T, Z_S, bw_method=kde_bw)

    nu_range_klkl = np.geomspace(1e-2, 1, num=11)
    nu_range_klhd = nu_range_klkl
    alpha_range = np.append(np.arange(0.05, 0.95 + 0.15, 0.15), 1.0)
    gamma_range = np.arange(-1, 3 + 0.5, step=0.5)
    # The conditional risk is [0, 4], but for this data sets it is usually <= 0.65.
    nu_range_hdhd = 4 / np.arange(1, 7, step = 1) + 1e-6  # [4, 2, 1.33, 1, 0.8, 0.67]

    # ==========================================
    # Phase 4: Model Training and Evaluation
    # ==========================================
    result_meta = {
      "replication": r,
      "noise_type": args.noise_type,
      "noise_sd": args.noise_sd,
      "bandwidth": args.nw_bw,
      "kernel": args.nw_kernel,
      "lr_kde_bw": args.kde_bw
    }
    # Baseline Models
    log_reg = LogisticRegression(max_iter=1000).fit(Z_S, y_S)
    prob_log_reg = log_reg.predict_proba(Z_T)[:, 1]
    prob_log_reg_clean = log_reg.predict_proba(Z_T_clean)[:, 1]
    record_result(result_meta | evaluate("logistic_regression", y_T, prob_log_reg))
    record_result(result_meta | evaluate("logistic_regression_clean", y_T, prob_log_reg_clean))

    svm_model = SVC(probability=True, kernel='rbf').fit(Z_S, y_S)
    prob_svm = svm_model.predict_proba(Z_T)[:, 1]
    prob_svm_clean = svm_model.predict_proba(Z_T_clean)[:, 1]
    record_result(result_meta | evaluate("svm", y_T, prob_svm))
    record_result(result_meta | evaluate("svm_clean", y_T, prob_svm_clean))

    nw_train = WeightedNadarayaWatson(Z_S, bandwidth=args.nw_bw, kernel=args.nw_kernel, clip=(0, 1))
    nw_test = WeightedNadarayaWatson(Z_S, X_test=Z_T, bandwidth=args.nw_bw, kernel=args.nw_kernel, clip=(0, 1))
    nw_clean = WeightedNadarayaWatson(Z_S, X_test=Z_T_clean, bandwidth=args.nw_bw, kernel=args.nw_kernel, clip=(0, 1))

    # KLIEP weights (fit in a subprocess; isolates TensorFlow from the main torch process)
    try:
      seed = int(rng.integers(0, 2**32, dtype=np.uint64))
      w_kliep = _run_reweighting_subprocess(_kliep_worker, Z_S, y_S, Z_T, seed)
      prob_nw_kliep = nw_test.predict(y_S, w_kliep)
      record_result(result_meta | evaluate("nw_kliep", y_T, prob_nw_kliep))
    except Exception as e:
      print(f"Error occurred while calculating NW Kliep: {e}")

    # uLSIF weights (fit in a subprocess; isolates TensorFlow from the main torch process)
    try:
      seed = int(rng.integers(0, 2**32, dtype=np.uint64))
      w_ulsif = _run_reweighting_subprocess(_ulsif_worker, Z_S, y_S, Z_T, seed)
      prob_nw_ulsif = nw_test.predict(y_S, w_ulsif)
      record_result(result_meta | evaluate("nw_ulsif", y_T, prob_nw_ulsif))
    except Exception as e:
      print(f"Error occurred while calculating NW ULSIF: {e}")

    prob_nw_src = nw_test.predict(y_S)
    record_result(result_meta | evaluate("nw_src", y_T, prob_nw_src))
    prob_nw_src_clean = nw_clean.predict(y_S)
    record_result(result_meta | evaluate("nw_src_clean", y_T, prob_nw_src_clean))

    for alpha in alpha_range:
      print(f"\nCalculating bridged NW weights for alpha={alpha:.2f}...")
      w_klkl = get_bridge_weights(lr_kde, geometry='kl_kl', alpha=alpha)
      w_hdhd = get_bridge_weights(lr_kde, geometry='hd_hd', alpha=alpha)

      prob_nw_klkl = nw_test.predict(y_S, w_klkl)
      record_result(result_meta | evaluate("nw_klkl", y_T, prob_nw_klkl, alpha))
      prob_nw_hdhd = nw_test.predict(y_S, w_hdhd)
      record_result(result_meta | evaluate("nw_hdhd", y_T, prob_nw_hdhd, alpha))

      if alpha < 1.0:
        for gamma in gamma_range:
          print(f"\nCalculating bridged KL/HD NW weights for alpha={alpha:.2f} and gamma={gamma:.2f}...")
          w_klhd = get_bridge_weights(lr_kde, geometry='kl_hd', alpha=alpha, gamma=gamma)
          prob_nw_klhd = nw_test.predict(y_S, w_klhd)
          record_result(result_meta | evaluate("nw_klhd", y_T, prob_nw_klhd, alpha, gamma))

      # Loss-adversarial KL/KL NW models
      for nu in nu_range_klkl:
        print(f"\nCalculating loss-adversarial KL/KL NW weights for alpha={alpha:.2f} and nu={nu:.2f}...")
        prob_nw_adv_klkl, _ = fit_loss_adversarial_nw(nw_train, nw_test,
                                                      y_S, lr_kde,
                                                      geometry='kl_kl',
                                                      loss='cross_entropy',
                                                      alpha=alpha,
                                                      nu=nu,
                                                      max_iter=_ADV_MAX_ITER,
                                                      iter_conv_tol=_ADV_ITER_CONV_TOL,
                                                      step_size=_ADV_STEP_SIZE)
        record_result(result_meta | evaluate("nw_adv_klkl", y_T, prob_nw_adv_klkl, alpha, nu=nu))

      # Loss-adversarial HD/HD NW models
      for nu in nu_range_hdhd:
        print(f"\nCalculating loss-adversarial HD/HD NW weights for alpha={alpha:.2f} and nu={nu:.2f}...")
        try:
          prob_nw_adv_hdhd, _ = fit_loss_adversarial_nw(nw_train, nw_test,
                                                        y_S, lr_kde,
                                                        geometry='hd_hd',
                                                        loss='cross_entropy',
                                                        alpha=alpha,
                                                        nu=nu,
                                                        max_iter=_ADV_MAX_ITER,
                                                        iter_conv_tol=_ADV_ITER_CONV_TOL,
                                                        step_size=_ADV_STEP_SIZE)
          record_result(result_meta | evaluate("nw_adv_hdhd", y_T, prob_nw_adv_hdhd, alpha, nu=nu))
        except ValueError as e:
          print(f"Skipping loss-adversarial HD/HD NW for alpha={alpha:.2f} and nu={nu:.2f} due to error: {e}", file=sys.stderr, flush=True)
          continue

      if alpha < 1.0:
        # Loss-adversarial KL/HD NW models
        for nu in nu_range_klhd:
          for gamma in gamma_range:
            print(f"\nCalculating loss-adversarial KL/HD NW weights for alpha={alpha:.2f}, gamma={gamma:.2f}, nu={nu:.2f}...")
            prob_nw_adv_klhd, _ = fit_loss_adversarial_nw(nw_train, nw_test,
                                                          y_S, lr_kde,
                                                          geometry='kl_hd',
                                                          loss='cross_entropy',
                                                          alpha=alpha,
                                                          gamma=gamma,
                                                          nu=nu,
                                                          max_iter=_ADV_MAX_ITER,
                                                          iter_conv_tol=_ADV_ITER_CONV_TOL,
                                                          step_size=_ADV_STEP_SIZE)
            record_result(result_meta | evaluate("nw_adv_klhd", y_T, prob_nw_adv_klhd, alpha, gamma, nu))

if __name__ == "__main__":
  main()
