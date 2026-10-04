# logistic_fedavg/server_app.py

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from LogesticRegression_comprehensive.task import (
    replace_keys,
    load_test_for_site_scaled,
    load_train_for_site_scaled,
    load_val_for_site_scaled,
    get_feature_columns,
    get_site_split_scaled_cached,
    sigmoid,
    binary_cross_entropy,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", category=UserWarning)

app = ServerApp()

EXPLAIN_RESULTS_DIR = Path("explainability_outputs")

SHAP_BACKGROUND_MAX = 30
SHAP_EXPLAIN_MAX = 80

LIME_NUM_CASES = 80
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 1000



def exp_tag(cfg) -> str:
    cancer = str(cfg["data"].get("cancer_name", "pancreas")).strip().lower()
    stage = str(cfg["data"].get("stage_name", "distant")).strip().lower()
    return f"{cancer}_{stage}"


# =========================
# Utilities
# =========================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_log(out_dir: Path, filename: str, message: str) -> None:
    _ensure_dir(out_dir)
    with open(out_dir / filename, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _sample_rows(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def _stabilize_lime_background(
    X_background: np.ndarray,
    X_explain: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    X_background = np.asarray(X_background, dtype=np.float32).copy()
    X_explain = np.asarray(X_explain, dtype=np.float32).copy()

    if X_background.ndim != 2:
        raise ValueError(f"X_background must be 2D, got shape={X_background.shape}")
    if X_explain.ndim != 2:
        raise ValueError(f"X_explain must be 2D, got shape={X_explain.shape}")

    stds = np.std(X_background, axis=0)
    bad_mask = (~np.isfinite(stds)) | (stds <= eps)

    if np.any(bad_mask):
        bad_idx = np.where(bad_mask)[0]

        for j in bad_idx:
            col = X_background[:, j]
            finite_col = col[np.isfinite(col)]
            base = float(finite_col.mean()) if len(finite_col) > 0 else 0.0

            noise = np.linspace(-1e-4, 1e-4, num=len(col), dtype=np.float32)
            X_background[:, j] = base + noise

            if X_explain.shape[0] > 0:
                X_explain[:, j] = base

    return X_background, X_explain


# =========================
# Bootstrap CI (NO MEAN)
# =========================
def _bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, alpha=0.95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    idx = np.arange(n)

    stats = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        yt = y_true[sample]
        yp = y_prob[sample]
        try:
            stats.append(metric_fn(yt, yp))
        except Exception:
            continue

    if len(stats) < max(30, int(n_boot * 0.2)):
        return float("nan"), float("nan")

    stats = np.asarray(stats, dtype=float)
    lo = float(np.quantile(stats, (1 - alpha) / 2))
    hi = float(np.quantile(stats, 1 - (1 - alpha) / 2))
    return lo, hi


def _format_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "nan"
    return f"[{lo:.4f}, {hi:.4f}]"


# =========================
# Point Metrics
# =========================
def _compute_point_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    y_prob_safe = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "log_loss": float(binary_cross_entropy(y_true.astype(np.float32), y_prob_safe.astype(np.float32), eps=1e-12)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# =========================
# Metrics + CI (NO MEAN)
# =========================
def classification_report_with_ci(y_true, y_prob, thr=0.5, n_boot=2000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    rep = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

    rep["accuracy_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: accuracy_score(yt, (yp >= thr).astype(int)),
        n_boot=n_boot, seed=seed
    )
    rep["precision_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: precision_score(yt, (yp >= thr).astype(int), zero_division=0),
        n_boot=n_boot, seed=seed
    )
    rep["recall_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: recall_score(yt, (yp >= thr).astype(int), zero_division=0),
        n_boot=n_boot, seed=seed
    )
    rep["f1_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: f1_score(yt, (yp >= thr).astype(int), zero_division=0),
        n_boot=n_boot, seed=seed
    )
    rep["roc_auc_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
        n_boot=n_boot, seed=seed
    )
    rep["pr_auc_ci95"] = _bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: average_precision_score(yt, yp),
        n_boot=n_boot, seed=seed
    )

    return rep


# =========================
# Pretty Print
# =========================
def print_one_line(tag, title, rep):
    def ci_str(ci):
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    print(
        f"{tag:<6} | {title:<35}: "
        f"ACC={rep['accuracy']:.4f}  ACC_CI={ci_str(rep['accuracy_ci95'])}  "
        f"PREC={rep['precision']:.4f}  PREC_CI={ci_str(rep['precision_ci95'])}  "
        f"REC={rep['recall']:.4f}  REC_CI={ci_str(rep['recall_ci95'])}  "
        f"F1={rep['f1']:.4f}  F1_CI={ci_str(rep['f1_ci95'])}  "
        f"ROC-AUC={rep['roc_auc']:.4f}  ROC-AUC_CI={ci_str(rep['roc_auc_ci95'])}  "
        f"PR-AUC={rep['pr_auc']:.4f}  PR-AUC_CI={ci_str(rep['pr_auc_ci95'])}"
    )



def _predict_proba_lr(w: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.clip(sigmoid(X @ w + b[0]), 1e-7, 1 - 1e-7)


def _predict_proba_two_col_lr(w: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    p1 = _predict_proba_lr(w, b, X)
    p0 = 1.0 - p1
    return np.column_stack([p0, p1])


# =========================
# Local Training
# =========================
def _fit_local_model_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    local_epochs: int,
    batch_size: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    w = np.zeros(d, dtype=np.float32)
    b = np.zeros(1, dtype=np.float32)

    batch_size = max(1, min(int(batch_size), n))

    best_w = w.copy()
    best_b = b.copy()
    best_val_loss = float("inf")

    for _ in range(int(local_epochs)):
        perm = rng.permutation(n)
        Xp = X_train[perm]
        yp = y_train[perm]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = Xp[start:end]
            yb = yp[start:end]

            prob = sigmoid(xb @ w + b[0])
            err = prob - yb

            grad_w = (xb.T @ err) / len(xb) + l2 * w
            grad_b = np.array([np.mean(err)], dtype=np.float32)

            w = w - lr * grad_w.astype(np.float32)
            b = b - lr * grad_b.astype(np.float32)

        val_prob = sigmoid(X_val @ w + b[0])
        val_loss = binary_cross_entropy(y_val, val_prob, eps=1e-12) + 0.5 * l2 * float(np.sum(w ** 2))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_w = w.copy()
            best_b = b.copy()

    return best_w, best_b


def _save_convergence_plot(df_rounds: pd.DataFrame, out_path: str = "convergence_plot.png") -> None:
    if df_rounds.empty:
        return

    plt.figure(figsize=(12, 8))

    metrics_to_plot = [
        ("auc", "AUC"),
        ("log_loss", "Log Loss"),
    ]

    for col, label in metrics_to_plot:
        if col in df_rounds.columns:
            plt.plot(df_rounds["round"], df_rounds[col], marker="o", label=label)

    plt.xlabel("Federated Round")
    plt.ylabel("Metric Value")
    plt.title("Federated Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================
# SHAP
# =========================
def _run_shap_for_model(
    w: np.ndarray,
    b: np.ndarray,
    feature_names: List[str],
    X_train_background: np.ndarray,
    X_test_explain: np.ndarray,
    seed: int,
    out_dir: Path,
    prefix: str,
    exp_name: str,
) -> None:
    _ensure_dir(out_dir)

    try:
        X_background = _sample_rows(X_train_background, SHAP_BACKGROUND_MAX, seed)
        X_explain = _sample_rows(X_test_explain, SHAP_EXPLAIN_MAX, seed + 1)

        def predict_fn(x):
            x = np.asarray(x, dtype=np.float32)
            return _predict_proba_lr(w, b, x)

        explainer = shap.Explainer(predict_fn, X_background)
        shap_values_obj = explainer(X_explain)

        shap_values = np.asarray(shap_values_obj.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        if shap_values.shape[-1] != len(feature_names):
            raise ValueError(
                f"SHAP shape mismatch: shap_values.shape={shap_values.shape}, "
                f"n_features={len(feature_names)}"
            )

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(out_dir / f"shap_values_{prefix}_{exp_name}.csv", index=False)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=X_explain,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            out_dir / f"shap_summary_beeswarm_{prefix}_{exp_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved SHAP outputs for {prefix}")

    except Exception as e:
        _append_log(out_dir, "shap_error_log.txt", f"SHAP failed for {prefix}: {repr(e)}")


# =========================
# LIME
# =========================
def _aggregate_lime_signed_importance(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    seed: int,
    num_cases: int = LIME_NUM_CASES,
    num_features: int = LIME_NUM_FEATURES,
    num_samples: int = LIME_NUM_SAMPLES,
) -> pd.Series:
    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)

    X_background, X_explain = _stabilize_lime_background(
        X_background,
        X_explain,
        eps=1e-8,
    )

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    if len(X_explain) > num_cases:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_explain), size=num_cases, replace=False)
        X_sel = X_explain[idx]
    else:
        X_sel = X_explain

    weight_map = {f: [] for f in feature_names}

    for row in X_sel:
        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=predict_proba_fn,
            num_features=min(num_features, len(feature_names)),
            num_samples=num_samples,
        )

        local_list = exp.as_list(label=1)

        for feat_desc, weight in local_list:
            matched_feature = None
            for fname in feature_names:
                if fname in feat_desc:
                    matched_feature = fname
                    break
            if matched_feature is not None:
                weight_map[matched_feature].append(float(weight))

    signed_means = {}
    for fname in feature_names:
        vals = weight_map[fname]
        signed_means[fname] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return pd.Series(signed_means).sort_values()


def _plot_lime_like_paper(
    signed_importance: pd.Series,
    title: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    s = signed_importance.copy()
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
    s = s.sort_values()

    colors = ["green" if v < 0 else "red" for v in s.values]

    plt.figure(figsize=(8, 6))
    plt.barh(s.index, s.values, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("LIME Weight (Direction of Influence)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _run_lime_for_model(
    w: np.ndarray,
    b: np.ndarray,
    X_train_background: np.ndarray,
    X_test_explain: np.ndarray,
    feature_names: List[str],
    seed: int,
    out_dir: Path,
    prefix: str,
    exp_name: str,
) -> None:
    _ensure_dir(out_dir)

    try:
        X_background = _sample_rows(X_train_background, 300, seed)
        X_explain = _sample_rows(X_test_explain, 200, seed + 1)

        X_background, X_explain = _stabilize_lime_background(
            X_background=X_background,
            X_explain=X_explain,
            eps=1e-8,
        )

        predict_fn = lambda x: _predict_proba_two_col_lr(
            w,
            b,
            np.asarray(x, dtype=np.float32),
        )

        signed_importance = _aggregate_lime_signed_importance(
            predict_proba_fn=predict_fn,
            X_background=X_background,
            X_explain=X_explain,
            feature_names=feature_names,
            class_names=["Negative", "Positive"],
            seed=seed,
        )

        signed_importance.to_csv(
            out_dir / f"lime_signed_importance_{prefix}_{exp_name}.csv",
            header=["weight"],
        )

        _plot_lime_like_paper(
            signed_importance=signed_importance,
            title=f"LIME Feature Importance ({prefix})",
            out_path=out_dir / f"lime_feature_importance_{prefix}_{exp_name}.png",
            top_n=15,
        )

        print(f"Saved LIME outputs for {prefix}")

    except Exception as e:
        _append_log(out_dir, "lime_error_log.txt", f"LIME failed for {prefix}: {repr(e)}")



def _run_posthoc_explainability(
    cfg,
    fed_w: np.ndarray,
    fed_b: np.ndarray,
    local_models: List[tuple[np.ndarray, np.ndarray]],
    site_paths: List[str],
    experiment_tag: str,
) -> None:
    _ensure_dir(EXPLAIN_RESULTS_DIR)

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])

    feature_names = get_feature_columns(site_paths[0])

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for i, p in enumerate(site_paths):
        X_tr, y_tr = load_train_for_site_scaled(p, seed, test_size, val_size)
        X_te, y_te = load_test_for_site_scaled(p, seed, test_size, val_size)

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_test_all.append(X_te)
        y_test_all.append(y_te)

    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)
    X_test_all = np.concatenate(X_test_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)

    _run_shap_for_model(
        w=fed_w,
        b=fed_b,
        feature_names=feature_names,
        X_train_background=X_train_all,
        X_test_explain=X_test_all,
        seed=seed,
        out_dir=EXPLAIN_RESULTS_DIR,
        prefix="federated_global",
        exp_name=experiment_tag,
    )

    _run_lime_for_model(
        w=fed_w,
        b=fed_b,
        X_train_background=X_train_all,
        X_test_explain=X_test_all,
        feature_names=feature_names,
        seed=seed,
        out_dir=EXPLAIN_RESULTS_DIR,
        prefix="federated_global",
        exp_name=experiment_tag,
    )

    for i, p in enumerate(site_paths):
        X_tr, _ = load_train_for_site_scaled(p, seed, test_size, val_size)
        X_te, _ = load_test_for_site_scaled(p, seed, test_size, val_size)
        w_local, b_local = local_models[i]

        _run_shap_for_model(
            w=w_local,
            b=b_local,
            feature_names=feature_names,
            X_train_background=X_tr,
            X_test_explain=X_te,
            seed=seed + 100 + i,
            out_dir=EXPLAIN_RESULTS_DIR,
            prefix=f"local_site{i+1}",
            exp_name=experiment_tag,
        )

        _run_lime_for_model(
            w=w_local,
            b=b_local,
            X_train_background=X_tr,
            X_test_explain=X_te,
            feature_names=feature_names,
            seed=seed + 200 + i,
            out_dir=EXPLAIN_RESULTS_DIR,
            prefix=f"local_site{i+1}",
            exp_name=experiment_tag,
        )



@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = replace_keys(unflatten_dict(context.run_config))
    experiment_tag = exp_tag(cfg)

    site_paths = [
        cfg["data"]["site1_path"],
        cfg["data"]["site2_path"],
        cfg["data"]["site3_path"],
    ]

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])

    num_rounds = int(cfg["fl"]["num_server_rounds"])
    fraction_train = float(cfg["fl"]["fraction_train"])
    fraction_evaluate = float(cfg["fl"].get("fraction_evaluate", 1.0))

    local_epochs = int(cfg["fl"]["local_epochs"])
    batch_size = int(cfg["fl"]["batch_size"])

    thr = float(cfg["report"]["threshold"])
    n_boot = int(cfg["report"]["bootstrap_samples"])

    lr = float(cfg["params"]["lr"])
    l2 = float(cfg["params"].get("l2", 0.0))

    first_split = get_site_split_scaled_cached(site_paths[0], seed, test_size, val_size)
    input_dim = int(first_split.X_train.shape[1])

    initial_arrays = ArrayRecord(
        [
            np.zeros(input_dim, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]
    )

    X_val_all, y_val_all = [], []
    for p in site_paths:
        X_va, y_va = load_val_for_site_scaled(p, seed, test_size, val_size)
        X_val_all.append(X_va)
        y_val_all.append(y_va)
    X_val_all = np.concatenate(X_val_all, axis=0)
    y_val_all = np.concatenate(y_val_all, axis=0)

    round_history: list[dict] = []

    def server_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        y_prob = _predict_proba_lr(
            np.array(arrays["0"].numpy(), dtype=np.float32),
            np.array(arrays["1"].numpy(), dtype=np.float32),
            X_val_all,
        )
        metrics = _compute_point_metrics(y_val_all, y_prob, thr=thr)

        row = {
            "round": int(server_round),
            "auc": float(metrics["auc"]),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "log_loss": float(metrics["log_loss"]),
            "pr_auc": float(metrics["pr_auc"]),
        }
        round_history.append(row)

        pd.DataFrame(round_history).to_csv(f"federated_metrics_per_round_{experiment_tag}.csv", index=False)

        return MetricRecord({
            "auc": row["auc"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "log_loss": row["log_loss"],
            "pr_auc": row["pr_auc"],
            "num-examples": int(len(y_val_all)),
        })

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        evaluate_fn=server_evaluate,
    )

    df_rounds = pd.DataFrame(round_history)
    df_rounds.to_csv(f"federated_metrics_per_round_{experiment_tag}.csv", index=False)
    _save_convergence_plot(df_rounds, out_path=f"convergence_plot_{experiment_tag}.png")

    fed_arrays = result.arrays
    fed_w = np.array(fed_arrays["0"].numpy(), dtype=np.float32)
    fed_b = np.array(fed_arrays["1"].numpy(), dtype=np.float32)

    np.savez("final_model.npz", w=fed_w, b=fed_b)
    print("Saving final federated logistic model to disk.\n")

    local_models = []
    local_budget_epochs = num_rounds * local_epochs

    for i, p in enumerate(site_paths):
        X_tr, y_tr = load_train_for_site_scaled(p, seed, test_size, val_size)
        X_va, y_va = load_val_for_site_scaled(p, seed, test_size, val_size)

        w_local, b_local = _fit_local_model_numpy(
            X_train=X_tr,
            y_train=y_tr.astype(np.float32),
            X_val=X_va,
            y_val=y_va.astype(np.float32),
            lr=lr,
            local_epochs=local_budget_epochs,
            batch_size=batch_size,
            l2=l2,
            seed=seed + i,
        )

        local_models.append((w_local, b_local))
        print(f"[LOCAL] site={i+1}, trained_epochs={local_budget_epochs}")

    print("=" * 78)

    final_rows = []

    X_all, y_all = [], []
    for p in site_paths:
        X_te, y_te = load_test_for_site_scaled(p, seed, test_size, val_size)
        X_all.append(X_te)
        y_all.append(y_te)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    fed_prob_all = _predict_proba_lr(fed_w, fed_b, X_all)
    rep_all = classification_report_with_ci(y_all, fed_prob_all, thr, n_boot, seed)
    print_one_line("FED", "Final Model | Concat Test(all sites)", rep_all)

    final_rows.append({
        "scope": "FED | Concat Test(all sites)",
        "Accuracy": rep_all["accuracy"],
        "Accuracy(CI95%)": _format_ci(*rep_all["accuracy_ci95"]),
        "Precision": rep_all["precision"],
        "Precision(CI95%)": _format_ci(*rep_all["precision_ci95"]),
        "Recall": rep_all["recall"],
        "Recall(CI95%)": _format_ci(*rep_all["recall_ci95"]),
        "F1": rep_all["f1"],
        "F1(CI95%)": _format_ci(*rep_all["f1_ci95"]),
        "ROC_AUC": rep_all["roc_auc"],
        "ROC_AUC(CI95%)": _format_ci(*rep_all["roc_auc_ci95"]),
        "PR_AUC": rep_all["pr_auc"],
        "PR_AUC(CI95%)": _format_ci(*rep_all["pr_auc_ci95"]),
    })

    print("\nFEDERATED Model per-site test:")
    for i, p in enumerate(site_paths):
        X_te, y_te = load_test_for_site_scaled(p, seed, test_size, val_size)
        y_prob = _predict_proba_lr(fed_w, fed_b, X_te)
        rep = classification_report_with_ci(y_te, y_prob, thr, n_boot, seed + i)
        print_one_line("FED", f"Final Model | Test(site{i+1})", rep)

        final_rows.append({
            "scope": f"FED | Test(site{i+1})",
            "Accuracy": rep["accuracy"],
            "Accuracy(CI95%)": _format_ci(*rep["accuracy_ci95"]),
            "Precision": rep["precision"],
            "Precision(CI95%)": _format_ci(*rep["precision_ci95"]),
            "Recall": rep["recall"],
            "Recall(CI95%)": _format_ci(*rep["recall_ci95"]),
            "F1": rep["f1"],
            "F1(CI95%)": _format_ci(*rep["f1_ci95"]),
            "ROC_AUC": rep["roc_auc"],
            "ROC_AUC(CI95%)": _format_ci(*rep["roc_auc_ci95"]),
            "PR_AUC": rep["pr_auc"],
            "PR_AUC(CI95%)": _format_ci(*rep["pr_auc_ci95"]),
        })

    print("\nLOCAL Models per-site test:")
    for i, p in enumerate(site_paths):
        X_te, y_te = load_test_for_site_scaled(p, seed, test_size, val_size)
        w_local, b_local = local_models[i]
        y_prob = _predict_proba_lr(w_local, b_local, X_te)
        rep = classification_report_with_ci(y_te, y_prob, thr, n_boot, seed + 100 + i)
        print_one_line("LOCAL", f"Site {i+1} | Test(site{i+1})", rep)

        final_rows.append({
            "scope": f"LOCAL | Test(site{i+1})",
            "Accuracy": rep["accuracy"],
            "Accuracy(CI95%)": _format_ci(*rep["accuracy_ci95"]),
            "Precision": rep["precision"],
            "Precision(CI95%)": _format_ci(*rep["precision_ci95"]),
            "Recall": rep["recall"],
            "Recall(CI95%)": _format_ci(*rep["recall_ci95"]),
            "F1": rep["f1"],
            "F1(CI95%)": _format_ci(*rep["f1_ci95"]),
            "ROC_AUC": rep["roc_auc"],
            "ROC_AUC(CI95%)": _format_ci(*rep["roc_auc_ci95"]),
            "PR_AUC": rep["pr_auc"],
            "PR_AUC(CI95%)": _format_ci(*rep["pr_auc_ci95"]),
        })

    pd.DataFrame(final_rows).to_csv(f"final_federated_metrics_{experiment_tag}.csv", index=False)

    _run_posthoc_explainability(
        cfg=cfg,
        fed_w=fed_w,
        fed_b=fed_b,
        local_models=local_models,
        site_paths=site_paths,
        experiment_tag=experiment_tag,
    )

    print("=" * 78)
    print(f"Saved: federated_metrics_per_round_{experiment_tag}.csv")
    print(f"Saved: final_federated_metrics_{experiment_tag}.csv")
    print(f"Saved: convergence_plot_{experiment_tag}.png")
    print("Saved: final_model.npz")
    print(f"Saved explainability outputs in: {EXPLAIN_RESULTS_DIR.resolve()}")
