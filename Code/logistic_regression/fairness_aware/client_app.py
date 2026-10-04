# logistic_fedavg/client_app.py

from __future__ import annotations

import warnings
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from LogesticRegression_comprehensive.task import (
    load_data_for_client_scaled,
    replace_keys,
    get_site_split_scaled_cached,
    get_kfold_train_val_splits_scaled,   # ✅ اضافه شد
    sigmoid,
    binary_cross_entropy,
    roc_auc_safe,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


def _init_params(input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros(input_dim, dtype=np.float32)
    b = np.zeros(1, dtype=np.float32)
    return w, b


def _train_local_logreg(
    X_train,
    y_train,
    X_val,
    y_val,
    w,
    b,
    lr,
    local_epochs,
    batch_size,
    l2,
    seed,
):
    rng = np.random.default_rng(seed)

    n = X_train.shape[0]
    batch_size = max(1, min(int(batch_size), n))

    w = w.copy()
    b = b.copy()

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

            logits = xb @ w + b[0]
            probs = sigmoid(logits)

            error = probs - yb
            grad_w = (xb.T @ error) / len(xb) + l2 * w
            grad_b = np.array([np.mean(error)], dtype=np.float32)

            w = w - lr * grad_w.astype(np.float32)
            b = b - lr * grad_b.astype(np.float32)

        val_prob = sigmoid(X_val @ w + b[0])
        val_loss = binary_cross_entropy(y_val, val_prob, eps=1e-12)
        val_loss = val_loss + 0.5 * l2 * float(np.sum(w ** 2))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_w = w.copy()
            best_b = b.copy()

    val_prob = sigmoid(X_val @ best_w + best_b[0])
    val_auc = roc_auc_safe(y_val, val_prob)

    return best_w, best_b, {
        "val-loss": float(binary_cross_entropy(y_val, val_prob, eps=1e-12)),
        "val-auc": float(val_auc),
    }


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    cfg = replace_keys(unflatten_dict(context.run_config))

    site_paths = [
        cfg["data"]["site1_path"],
        cfg["data"]["site2_path"],
        cfg["data"]["site3_path"],
    ]

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])
    k_folds = int(cfg["data"].get("k_folds", 5))  # ✅

    local_epochs = int(cfg["fl"]["local_epochs"])
    batch_size = int(cfg["fl"]["batch_size"])
    lr = float(cfg["params"]["lr"])
    l2 = float(cfg["params"].get("l2", 0.0))

    server_round = int(msg.content["config"]["server-round"])

    X_train, y_train, X_val, y_val, n_train, _ = load_data_for_client_scaled(
        site_paths=site_paths,
        partition_id=partition_id,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    input_dim = X_train.shape[1]

    if server_round <= 1 or "arrays" not in msg.content:
        global_w, global_b = _init_params(input_dim)
    else:
        global_w = np.array(msg.content["arrays"]["0"].numpy(), dtype=np.float32)
        global_b = np.array(msg.content["arrays"]["1"].numpy(), dtype=np.float32)

        if global_w.shape[0] != input_dim:
            global_w, global_b = _init_params(input_dim)

    # =========================
    # 1) K-Fold (ONLY evaluation)
    # =========================
    folds = get_kfold_train_val_splits_scaled(
        site_path=site_paths[partition_id],
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        k_folds=k_folds,
    )

    fold_losses = []
    fold_aucs = []

    for fold_id, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
        _, _, metrics = _train_local_logreg(
            X_train=X_tr,
            y_train=y_tr.astype(np.float32),
            X_val=X_va,
            y_val=y_va.astype(np.float32),
            w=global_w.copy(),
            b=global_b.copy(),
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
            l2=l2,
            seed=seed + partition_id * 1000 + server_round * 100 + fold_id,
        )

        fold_losses.append(metrics["val-loss"])
        fold_aucs.append(metrics["val-auc"])

    cv_val_loss = float(np.nanmean(fold_losses))
    cv_val_auc = float(np.nanmean(fold_aucs))

    # =========================
    # 2) REAL training (FedAvg)
    # =========================
    final_w, final_b, holdout_metrics = _train_local_logreg(
        X_train=X_train,
        y_train=y_train.astype(np.float32),
        X_val=X_val,
        y_val=y_val.astype(np.float32),
        w=global_w.copy(),
        b=global_b.copy(),
        lr=lr,
        local_epochs=local_epochs,
        batch_size=batch_size,
        l2=l2,
        seed=seed + partition_id + server_round,
    )

    arrays = ArrayRecord([final_w.astype(np.float32), final_b.astype(np.float32)])

    metrics = MetricRecord(
        {
            "num-examples": int(n_train),
            "cv-val-loss": cv_val_loss,     # ✅ جدید
            "cv-val-auc": cv_val_auc,       # ✅ جدید
            "val-loss": float(holdout_metrics["val-loss"]),
            "val-auc": float(holdout_metrics["val-auc"]),
        }
    )

    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])

    cfg = replace_keys(unflatten_dict(context.run_config))
    site_paths = [
        cfg["data"]["site1_path"],
        cfg["data"]["site2_path"],
        cfg["data"]["site3_path"],
    ]

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])

    split = get_site_split_scaled_cached(
        site_paths[partition_id],
        seed,
        test_size,
        val_size,
    )

    w = np.array(msg.content["arrays"]["0"].numpy(), dtype=np.float32)
    b = np.array(msg.content["arrays"]["1"].numpy(), dtype=np.float32)

    val_prob = sigmoid(split.X_val @ w + b[0])
    val_loss = binary_cross_entropy(split.y_val.astype(np.float32), val_prob, eps=1e-12)
    val_auc = roc_auc_safe(split.y_val, val_prob)

    metric_record = MetricRecord(
        {
            "loss": float(val_loss),
            "auc": float(val_auc),
            "num-examples": int(split.X_val.shape[0]),
        }
    )

    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)