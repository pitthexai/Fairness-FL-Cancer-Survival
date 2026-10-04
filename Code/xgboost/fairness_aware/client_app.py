# xgboost_comprehensive/client_app.py

from __future__ import annotations

import warnings
import numpy as np
import xgboost as xgb

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from xgboost_comprehensive.task import (
    load_data_for_client_scaled,
    replace_keys,
    get_kfold_train_val_splits_scaled,
    to_dmatrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


def _prepare_xgb_params(cfg) -> dict:
    params = dict(cfg["params"])

    if "reg_lambda" in params and "lambda" not in params:
        params["lambda"] = params.pop("reg_lambda")
    if "reg_alpha" in params and "alpha" not in params:
        params["alpha"] = params.pop("reg_alpha")

    return params


def _roc_auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _log_loss_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import log_loss
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def _copy_booster(bst: xgb.Booster | None, params: dict) -> xgb.Booster | None:
    if bst is None:
        return None

    copied_bst = xgb.Booster(params=params)
    copied_bst.load_model(bytearray(bst.save_raw("json")))
    return copied_bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    cfg = replace_keys(unflatten_dict(context.run_config))

    site_paths = [
        cfg["data"]["site1_path"],
        cfg["data"]["site2_path"],
        cfg["data"]["site3_path"],
    ]

    seed = int(cfg["data"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])
    k_folds = int(cfg["data"].get("k_folds", 5))

    num_local_round = int(cfg["fl"]["local_boost_rounds"])
    train_method = cfg["fl"]["train_method"]

    params = _prepare_xgb_params(cfg)

    if bool(cfg["fl"]["scaled_lr"]):
        params = dict(params)
        params["eta"] = float(params["eta"]) / max(1, num_partitions)

    dtrain, dval, n_train, _ = load_data_for_client_scaled(
        site_paths=site_paths,
        partition_id=partition_id,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    server_round = int(msg.content["config"]["server-round"])

    prev_bst = None
    prev_rounds = 0

    if server_round > 1:
        prev_bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        prev_bst.load_model(global_model)
        prev_rounds = prev_bst.num_boosted_rounds()


    folds = get_kfold_train_val_splits_scaled(
        site_path=site_paths[partition_id],
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        k_folds=k_folds,
    )

    fold_losses = []
    fold_aucs = []

    for fold_id, (X_train_f, y_train_f, X_val_f, y_val_f) in enumerate(folds):
        dtrain_f = to_dmatrix(X_train_f, y_train_f)
        dval_f = to_dmatrix(X_val_f, y_val_f)

        fold_bst = xgb.train(
            params=params,
            dtrain=dtrain_f,
            num_boost_round=num_local_round,
            evals=[(dval_f, "val")],
            xgb_model=_copy_booster(prev_bst, params),
            verbose_eval=False,
        )

        fold_preds = fold_bst.predict(dval_f)

        fold_losses.append(_log_loss_safe(y_val_f, fold_preds))
        fold_aucs.append(_roc_auc_safe(y_val_f, fold_preds))

    cv_val_loss = float(np.nanmean(fold_losses))
    cv_val_auc = float(np.nanmean(fold_aucs))

    # ------------------------------------------------------------
    # 2) Training for federated XGBoost
    # ------------------------------------------------------------

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_local_round,
        evals=[(dval, "val")],
        xgb_model=prev_bst,
        verbose_eval=False,
    )

    holdout_preds = bst.predict(dval)
    holdout_val_loss = _log_loss_safe(dval.get_label(), holdout_preds)
    holdout_val_auc = _roc_auc_safe(dval.get_label(), holdout_preds)

    total_rounds_after = bst.num_boosted_rounds()
    added_rounds = max(1, total_rounds_after - prev_rounds)

    if train_method == "bagging":
        bst_to_send = bst[total_rounds_after - added_rounds: total_rounds_after]
    else:
        bst_to_send = bst

    local_model = bst_to_send.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    metrics = {
        "num-examples": int(n_train),
        "added-boost-rounds": int(added_rounds),
        "cv-val-loss": cv_val_loss,
        "cv-val-auc": cv_val_auc,
        "holdout-val-loss": float(holdout_val_loss),
        "holdout-val-auc": float(holdout_val_auc),
    }

    model_record = ArrayRecord([model_np])
    metric_record = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metric_record})
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

    params = _prepare_xgb_params(cfg)

    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)


    _, dval, _, n_val = load_data_for_client_scaled(
        site_paths=site_paths,
        partition_id=partition_id,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    y_val = dval.get_label()
    val_preds = bst.predict(dval)

    val_loss = _log_loss_safe(y_val, val_preds)
    val_auc = _roc_auc_safe(y_val, val_preds)

    metric_record = MetricRecord(
        {
            "loss": float(val_loss),
            "auc": float(val_auc),
            "num-examples": int(n_val),
        }
    )

    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
