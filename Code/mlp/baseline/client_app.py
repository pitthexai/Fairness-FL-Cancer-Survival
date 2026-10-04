# MLP/mlp_client_app.py

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from mlp_comprehensive.task import (
    replace_keys,
    load_train_for_site_scaled,
    load_val_for_site_scaled,
    get_kfold_train_val_splits_scaled,
)
from mlp_comprehensive.models import MLP, get_weights, set_weights

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


def _make_site_paths(cfg) -> List[str]:
    return [
        cfg["data"]["site1_path"],
        cfg["data"]["site2_path"],
        cfg["data"]["site3_path"],
    ]


def _build_model(cfg, input_dim: int) -> MLP:
    hidden_dims = cfg.get("params", {}).get("hidden_dims", None)
    dropout = float(cfg.get("params", {}).get("dropout", 0.0))
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _bce_np(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float32), eps, 1.0 - eps)
    return float(
        -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )
    )


def _roc_auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except Exception:
        return float("nan")


def _predict_proba(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str,
) -> np.ndarray:
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(X_t).detach().cpu().numpy().reshape(-1)

    return np.clip(_sigmoid_np(logits), 1e-7, 1 - 1e-7)


def _train_one_client(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    local_epochs: int,
    weight_decay: float,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model.to(device)

    Xtr = torch.from_numpy(X_train.astype(np.float32))
    ytr = torch.from_numpy(y_train.astype(np.float32))

    batch_size = max(1, min(int(batch_size), len(Xtr)))

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    best_val_loss = float("inf")

    for _ in range(int(local_epochs)):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        val_prob = _predict_proba(model, X_val, device=device)
        val_loss = _bce_np(y_val, val_prob)

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state, strict=True)

    val_prob = _predict_proba(model, X_val, device=device)

    return {
        "val-loss": float(_bce_np(y_val, val_prob)),
        "val-auc": float(_roc_auc_safe(y_val, val_prob)),
    }


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    cfg = replace_keys(unflatten_dict(context.run_config))
    site_paths = _make_site_paths(cfg)

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])
    k_folds = int(cfg["data"].get("k_folds", 5))

    local_epochs = int(cfg["fl"].get("local_epochs", 1))

    lr = float(cfg.get("params", {}).get("lr", 1e-3))
    batch_size = int(cfg.get("params", {}).get("batch_size", 128))
    weight_decay = float(cfg.get("params", {}).get("weight_decay", 0.0))

    if bool(cfg["fl"].get("scaled_lr", False)):
        lr = lr / max(1, num_partitions)

    device = str(cfg.get("params", {}).get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    server_round = int(msg.content["config"]["server-round"])

    site_path = site_paths[partition_id]

    X_tr, y_tr = load_train_for_site_scaled(site_path, seed, test_size, val_size)
    X_va, y_va = load_val_for_site_scaled(site_path, seed, test_size, val_size)

    input_dim = X_tr.shape[1]

    global_model = _build_model(cfg, input_dim=input_dim)

    if server_round > 1 and "arrays" in msg.content:
        arrays = msg.content["arrays"]
        incoming = [arrays[str(i)].numpy() for i in range(len(arrays))]
        set_weights(global_model, incoming)

    global_weights = get_weights(global_model)

    # ------------------------------------------------------------
    # 1) K-Fold evaluation only
    # ------------------------------------------------------------
    folds = get_kfold_train_val_splits_scaled(
        site_path=site_path,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        k_folds=k_folds,
    )

    fold_losses = []
    fold_aucs = []

    for fold_id, (X_train_f, y_train_f, X_val_f, y_val_f) in enumerate(folds):
        fold_model = _build_model(cfg, input_dim=input_dim)
        set_weights(fold_model, global_weights)

        fold_metrics = _train_one_client(
            model=fold_model,
            X_train=X_train_f,
            y_train=y_train_f,
            X_val=X_val_f,
            y_val=y_val_f,
            lr=lr,
            batch_size=batch_size,
            local_epochs=local_epochs,
            weight_decay=weight_decay,
            device=device,
            seed=seed + partition_id * 1000 + server_round * 100 + fold_id,
        )

        fold_losses.append(fold_metrics["val-loss"])
        fold_aucs.append(fold_metrics["val-auc"])

    cv_val_loss = float(np.nanmean(fold_losses))
    cv_val_auc = float(np.nanmean(fold_aucs))

    # ------------------------------------------------------------
    # 2) Training for FedAvg
    # ------------------------------------------------------------
    final_model = _build_model(cfg, input_dim=input_dim)
    set_weights(final_model, global_weights)

    holdout_metrics = _train_one_client(
        model=final_model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        lr=lr,
        batch_size=batch_size,
        local_epochs=local_epochs,
        weight_decay=weight_decay,
        device=device,
        seed=seed + partition_id * 1000 + server_round,
    )

    weights = get_weights(final_model)

    model_record = ArrayRecord(weights)
    metric_record = MetricRecord(
        {
            "num-examples": int(X_tr.shape[0]),
            "cv-val-loss": cv_val_loss,
            "cv-val-auc": cv_val_auc,
            "holdout-val-loss": float(holdout_metrics["val-loss"]),
            "holdout-val-auc": float(holdout_metrics["val-auc"]),
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])

    cfg = replace_keys(unflatten_dict(context.run_config))
    site_paths = _make_site_paths(cfg)

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])
    k_folds = int(cfg["data"].get("k_folds", 5))

    device = str(cfg.get("params", {}).get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    site_path = site_paths[partition_id]

    folds = get_kfold_train_val_splits_scaled(
        site_path=site_path,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        k_folds=k_folds,
    )

    input_dim = folds[0][0].shape[1]

    model = _build_model(cfg, input_dim=input_dim)

    arrays = msg.content["arrays"]
    incoming = [arrays[str(i)].numpy() for i in range(len(arrays))]
    set_weights(model, incoming)

    all_y_val = []
    all_val_prob = []
    fold_losses = []

    for _, _, X_val_f, y_val_f in folds:
        val_prob = _predict_proba(model, X_val_f, device=device)
        val_loss = _bce_np(y_val_f, val_prob)

        fold_losses.append(val_loss)
        all_y_val.append(y_val_f)
        all_val_prob.append(val_prob)

    all_y_val = np.concatenate(all_y_val)
    all_val_prob = np.concatenate(all_val_prob)

    val_auc = _roc_auc_safe(all_y_val, all_val_prob)

    metric_record = MetricRecord(
        {
            "loss": float(np.nanmean(fold_losses)),
            "auc": float(val_auc),
            "num-examples": int(len(all_y_val)),
        }
    )

    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)