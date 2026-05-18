"""
Final Project - Step 5
Train a nonlinear fusion model (MLP) on tabular + image embeddings.

Purpose:
- Improve over linear fusion by allowing nonlinear interactions between
  tabular predictors and image-derived dimensions.
- Keep the pipeline modular and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from final_project_step1_tabular import Step1Config, build_preprocessor, get_feature_sets
from final_project_step3_fusion import (
    _to_dense_2d,
    build_image_block,
    hstack_tabular_image,
    load_embeddings_npz,
    load_splits,
    merge_tabular_with_embeddings,
)


@dataclass(frozen=True)
class Step5Config:
    step1_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    embeddings_dir: str = os.path.join("final_project_artifacts", "step2_image", "embeddings")
    step3_run_summary: str = os.path.join("final_project_artifacts", "step3_fusion", "run_summary.json")
    output_dir: str = os.path.join("final_project_artifacts", "step5_fusion_mlp")

    seed: int = 42
    batch_size: int = 128
    epochs: int = 120
    patience: int = 14
    lr: float = 1e-3
    weight_decay: float = 1e-4

    hidden1: int = 256
    hidden2: int = 96
    dropout: float = 0.25


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_best_step3_config(path: str) -> Tuple[str, str]:
    with open(path) as f:
        d = json.load(f)
    b = d["best_from_validation"]
    return str(b["tabular_feature_set"]), str(b["image_repr"])


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate(y_true_price: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = np.log1p(y_true_price)
    pred_price = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)
    return {
        "r2_log_price": float(r2_score(y_true_log, pred_log)),
        "rmse_log_price": _rmse(y_true_log, pred_log),
        "mae_log_price": float(mean_absolute_error(y_true_log, pred_log)),
        "r2_price": float(r2_score(y_true_price, pred_price)),
        "rmse_price": _rmse(y_true_price, pred_price),
        "mae_price": float(mean_absolute_error(y_true_price, pred_price)),
    }


def prepare_fused_features(
    cfg: Step5Config, tabular_fs: str, image_repr: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s1_cfg = Step1Config(seed=cfg.seed)
    feature_sets = get_feature_sets(s1_cfg)
    if tabular_fs not in feature_sets:
        raise ValueError(f"Unknown tabular_feature_set '{tabular_fs}'")

    train_df, val_df, test_df = load_splits(cfg.step1_dir)

    Xtr_e, idtr = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "train_resnet18_embeddings.npz"))
    Xva_e, idva = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "val_resnet18_embeddings.npz"))
    Xte_e, idte = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "test_resnet18_embeddings.npz"))

    train_df, Xtr_e = merge_tabular_with_embeddings(train_df, Xtr_e, idtr)
    val_df, Xva_e = merge_tabular_with_embeddings(val_df, Xva_e, idva)
    test_df, Xte_e = merge_tabular_with_embeddings(test_df, Xte_e, idte)

    fs_cols = feature_sets[tabular_fs]
    tabular_cols = [*fs_cols["numeric"], *fs_cols["categorical"]]
    pre = build_preprocessor(
        numeric_cols=list(fs_cols["numeric"]),
        categorical_cols=list(fs_cols["categorical"]),
    )
    T_tr = _to_dense_2d(pre.fit_transform(train_df[tabular_cols].copy()))
    T_va = _to_dense_2d(pre.transform(val_df[tabular_cols].copy()))
    T_te = _to_dense_2d(pre.transform(test_df[tabular_cols].copy()))

    I_tr, I_va, I_te, _ = build_image_block(image_repr, Xtr_e, Xva_e, Xte_e, cfg.seed)
    F_tr = hstack_tabular_image(T_tr, I_tr).astype(np.float32)
    F_va = hstack_tabular_image(T_va, I_va).astype(np.float32)
    F_te = hstack_tabular_image(T_te, I_te).astype(np.float32)

    y_tr = np.log1p(train_df["price"].to_numpy(dtype=np.float32))
    y_va = np.log1p(val_df["price"].to_numpy(dtype=np.float32))
    y_te_price = test_df["price"].to_numpy(dtype=np.float64)

    return F_tr, F_va, F_te, y_tr, y_va, y_te_price


def run(cfg: Step5Config, tabular_fs: str, image_repr: str) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)
    _set_seed(cfg.seed)

    F_tr, F_va, F_te, y_tr, y_va, y_te_price = prepare_fused_features(cfg, tabular_fs, image_repr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr) + 1e-8)
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_va_n = ((y_va - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(F_tr), torch.from_numpy(y_tr_n))
    val_ds = TensorDataset(torch.from_numpy(F_va), torch.from_numpy(y_va_n))
    test_x = torch.from_numpy(F_te).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = FusionMLP(
        in_dim=F_tr.shape[1],
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                va_losses.append(float(criterion(pred, yb).item()))

        row = {
            "epoch": epoch,
            "train_mse_log": float(np.mean(tr_losses)),
            "val_mse_log": float(np.mean(va_losses)),
        }
        history.append(row)

        if row["val_mse_log"] < best_val:
            best_val = row["val_mse_log"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint captured.")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_te_norm = model(test_x).cpu().numpy().astype(np.float64)
        pred_te_log = pred_te_norm * y_std + y_mean

    test_metrics = evaluate(y_te_price, pred_te_log)

    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump({"history": history}, f, indent=2)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "in_dim": int(F_tr.shape[1]),
            "hidden1": cfg.hidden1,
            "hidden2": cfg.hidden2,
            "dropout": cfg.dropout,
        },
        os.path.join(cfg.output_dir, "best_model.pt"),
    )

    preds_df = pd.DataFrame(
        {
            "y_true_price": y_te_price,
            "pred_log_price": pred_te_log,
            "pred_price": np.clip(np.expm1(pred_te_log), a_min=0.0, a_max=None),
        }
    )
    preds_df.to_csv(os.path.join(cfg.output_dir, "test_predictions.csv"), index=False)

    summary = {
        "config": asdict(cfg),
        "feature_selection": {"tabular_feature_set": tabular_fs, "image_repr": image_repr},
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "device": device.type,
    }
    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5: nonlinear fusion MLP")
    p.add_argument("--step1_dir", type=str, default=os.path.join("final_project_artifacts", "step1_tabular"))
    p.add_argument(
        "--embeddings_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step2_image", "embeddings"),
    )
    p.add_argument(
        "--step3_run_summary",
        type=str,
        default=os.path.join("final_project_artifacts", "step3_fusion", "run_summary.json"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step5_fusion_mlp"),
    )
    p.add_argument("--tabular_feature_set", type=str, default=None)
    p.add_argument("--image_repr", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=14)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tab_fs = args.tabular_feature_set
    img_repr = args.image_repr
    if tab_fs is None or img_repr is None:
        best_fs, best_img = load_best_step3_config(args.step3_run_summary)
        tab_fs = best_fs if tab_fs is None else tab_fs
        img_repr = best_img if img_repr is None else img_repr

    cfg = Step5Config(
        step1_dir=args.step1_dir,
        embeddings_dir=args.embeddings_dir,
        step3_run_summary=args.step3_run_summary,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    )
    summary = run(cfg, tabular_fs=tab_fs, image_repr=img_repr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

