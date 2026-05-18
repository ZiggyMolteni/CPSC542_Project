"""
Final Project - Step 5b
Explainability for the nonlinear fusion MLP (Step 5).

Focus:
1) Quantify image-block contribution with controlled masking/shuffling.
2) Estimate grouped permutation importance (numeric / city / image).
3) Test potential confounder: image brightness (day/night proxy).
4) Save residual cohorts for qualitative inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from final_project_step1_tabular import Step1Config, build_preprocessor, get_feature_sets
from final_project_step3_fusion import (
    _to_dense_2d,
    build_image_block,
    hstack_tabular_image,
    load_embeddings_npz,
    load_splits,
    merge_tabular_with_embeddings,
)
from final_project_step5_fusion_mlp import FusionMLP


@dataclass(frozen=True)
class Step5BConfig:
    step1_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    embeddings_dir: str = os.path.join("final_project_artifacts", "step2_image", "embeddings")
    step5_run_summary: str = os.path.join("final_project_artifacts", "step5_fusion_mlp", "run_summary.json")
    step5_ckpt: str = os.path.join("final_project_artifacts", "step5_fusion_mlp", "best_model.pt")
    output_dir: str = os.path.join("final_project_artifacts", "step5b_explain_mlp")
    image_dir: str = os.path.join("socal2", "socal_pics")
    seed: int = 42
    n_group_permutations: int = 16


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_step5_summary(path: str) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def _build_data(
    cfg: Step5BConfig, tabular_fs: str, image_repr: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    s1_cfg = Step1Config(seed=cfg.seed)
    feature_sets = get_feature_sets(s1_cfg)
    fs_cols = feature_sets[tabular_fs]
    tabular_cols = [*fs_cols["numeric"], *fs_cols["categorical"]]

    train_df, val_df, test_df = load_splits(cfg.step1_dir)
    Xtr_e, idtr = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "train_resnet18_embeddings.npz"))
    Xva_e, idva = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "val_resnet18_embeddings.npz"))
    Xte_e, idte = load_embeddings_npz(os.path.join(cfg.embeddings_dir, "test_resnet18_embeddings.npz"))

    train_df, Xtr_e = merge_tabular_with_embeddings(train_df, Xtr_e, idtr)
    val_df, Xva_e = merge_tabular_with_embeddings(val_df, Xva_e, idva)
    test_df, Xte_e = merge_tabular_with_embeddings(test_df, Xte_e, idte)

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

    y_tr_log = np.log1p(train_df["price"].to_numpy(dtype=np.float32))
    y_va_log = np.log1p(val_df["price"].to_numpy(dtype=np.float32))
    y_test_price = test_df["price"].to_numpy(dtype=np.float64)

    y_mean = float(np.mean(y_tr_log))
    y_std = float(np.std(y_tr_log) + 1e-8)
    y_trv_log = np.concatenate([y_tr_log, y_va_log]).astype(np.float64)

    n_tab = int(T_te.shape[1])
    trainval = np.vstack([F_tr, F_va])
    return trainval, F_te, y_trv_log, y_test_price, test_df.reset_index(drop=True), n_tab


def _predict_log(model: FusionMLP, X: np.ndarray, y_mean: float, y_std: float, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).to(device)
        out_n = model(t).cpu().numpy().astype(np.float64)
    return out_n * y_std + y_mean


def _r2_log(y_price: np.ndarray, pred_log: np.ndarray) -> float:
    return float(r2_score(np.log1p(y_price), pred_log))


def _mae_price(y_price: np.ndarray, pred_log: np.ndarray) -> float:
    pred_price = np.clip(np.expm1(pred_log), 0.0, None)
    return float(mean_absolute_error(y_price, pred_price))


def _brightness_for_image(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Luminance approximation
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(np.mean(lum))


def run(cfg: Step5BConfig) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)
    np.random.seed(cfg.seed)

    s5 = _load_step5_summary(cfg.step5_run_summary)
    tabular_fs = str(s5["feature_selection"]["tabular_feature_set"])
    image_repr = str(s5["feature_selection"]["image_repr"])
    model_cfg = s5["config"]

    trainval, X_test, y_trv_log, y_test_price, test_df, n_tab = _build_data(cfg, tabular_fs, image_repr)
    y_mean = float(np.mean(y_trv_log))
    y_std = float(np.std(y_trv_log) + 1e-8)

    ckpt = torch.load(cfg.step5_ckpt, map_location="cpu")
    model = FusionMLP(
        in_dim=int(ckpt["in_dim"]),
        hidden1=int(ckpt["hidden1"]),
        hidden2=int(ckpt["hidden2"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cpu")
    pred_full = _predict_log(model, X_test, y_mean, y_std, device)
    r2_full = _r2_log(y_test_price, pred_full)
    mae_full = _mae_price(y_test_price, pred_full)

    X_zero = X_test.copy()
    X_zero[:, n_tab:] = 0.0
    pred_zero = _predict_log(model, X_zero, y_mean, y_std, device)

    rng = np.random.default_rng(cfg.seed)
    X_shuffle = X_test.copy()
    shuf_idx = rng.permutation(len(X_shuffle))
    X_shuffle[:, n_tab:] = X_shuffle[shuf_idx, n_tab:]
    pred_shuffle = _predict_log(model, X_shuffle, y_mean, y_std, device)

    # Grouped permutation importance (numeric first 3 tab dims, remaining tab dims, image dims)
    groups = {
        "tabular_all": np.arange(0, n_tab),
        "image_block": np.arange(n_tab, X_test.shape[1]),
    }
    if n_tab >= 3:
        groups["tabular_numeric_head"] = np.arange(0, 3)
    if n_tab > 3:
        groups["tabular_city_like"] = np.arange(3, n_tab)

    group_rows: List[Dict[str, float]] = []
    for name, idxs in groups.items():
        drops = []
        for _ in range(cfg.n_group_permutations):
            Xp = X_test.copy()
            p = rng.permutation(len(Xp))
            Xp[:, idxs] = Xp[p][:, idxs]
            pred_p = _predict_log(model, Xp, y_mean, y_std, device)
            drops.append(r2_full - _r2_log(y_test_price, pred_p))
        group_rows.append(
            {
                "group": name,
                "mean_r2_log_drop": float(np.mean(drops)),
                "std_r2_log_drop": float(np.std(drops)),
            }
        )
    pd.DataFrame(group_rows).sort_values("mean_r2_log_drop", ascending=False).to_csv(
        os.path.join(cfg.output_dir, "group_permutation_importance.csv"), index=False
    )

    # Per-example image contribution proxy: full - zeroed-image predictions
    image_contrib_log = pred_full - pred_zero

    # Brightness/day-night proxy analysis
    brightness = []
    for image_id in test_df["image_id"].astype(int).tolist():
        p = os.path.join(cfg.image_dir, f"{image_id}.jpg")
        brightness.append(_brightness_for_image(p) if os.path.exists(p) else np.nan)
    brightness = np.array(brightness, dtype=np.float64)

    # Low/high brightness cohorts (quartiles)
    q1 = float(np.nanquantile(brightness, 0.25))
    q3 = float(np.nanquantile(brightness, 0.75))
    low_mask = brightness <= q1
    high_mask = brightness >= q3

    def cohort(mask: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "r2_log_full": _r2_log(y_test_price[mask], pred_full[mask]),
            "r2_log_zero_image": _r2_log(y_test_price[mask], pred_zero[mask]),
            "mean_image_contrib_log": float(np.nanmean(image_contrib_log[mask])),
            "mean_brightness": float(np.nanmean(brightness[mask])),
        }

    low_stats = cohort(low_mask)
    high_stats = cohort(high_mask)

    corr_brightness_pred = float(np.corrcoef(brightness, pred_full)[0, 1])
    corr_brightness_resid = float(np.corrcoef(brightness, np.log1p(y_test_price) - pred_full)[0, 1])
    corr_brightness_imgcontrib = float(np.corrcoef(brightness, image_contrib_log)[0, 1])

    out = test_df[["image_id", "price"]].copy()
    out["brightness_mean_luma"] = brightness
    out["pred_log_full"] = pred_full
    out["pred_log_zero_image"] = pred_zero
    out["image_contrib_log"] = image_contrib_log
    out["residual_log_full"] = np.log1p(out["price"].to_numpy()) - pred_full
    out["abs_residual_log_full"] = np.abs(out["residual_log_full"])
    out.to_csv(os.path.join(cfg.output_dir, "test_brightness_contrib_residuals.csv"), index=False)

    out.nlargest(25, "image_contrib_log").to_csv(
        os.path.join(cfg.output_dir, "examples_high_positive_image_contrib.csv"), index=False
    )
    out.nsmallest(25, "image_contrib_log").to_csv(
        os.path.join(cfg.output_dir, "examples_high_negative_image_contrib.csv"), index=False
    )
    out.nlargest(25, "abs_residual_log_full").to_csv(
        os.path.join(cfg.output_dir, "examples_worst_abs_residuals.csv"), index=False
    )

    summary = {
        "model_used": {
            "tabular_feature_set": tabular_fs,
            "image_repr": image_repr,
            "mlp_hidden1": model_cfg["hidden1"],
            "mlp_hidden2": model_cfg["hidden2"],
            "mlp_dropout": model_cfg["dropout"],
        },
        "metrics": {
            "r2_log_full": r2_full,
            "mae_price_full": mae_full,
            "r2_log_zero_image": _r2_log(y_test_price, pred_zero),
            "r2_log_shuffled_image": _r2_log(y_test_price, pred_shuffle),
            "delta_r2_full_minus_zero": r2_full - _r2_log(y_test_price, pred_zero),
            "delta_r2_full_minus_shuffled": r2_full - _r2_log(y_test_price, pred_shuffle),
        },
        "brightness_analysis": {
            "corr_brightness_vs_pred_log": corr_brightness_pred,
            "corr_brightness_vs_residual_log": corr_brightness_resid,
            "corr_brightness_vs_image_contrib_log": corr_brightness_imgcontrib,
            "low_brightness_q1_threshold": q1,
            "high_brightness_q3_threshold": q3,
            "low_brightness_stats": low_stats,
            "high_brightness_stats": high_stats,
        },
        "files": {
            "group_permutation_importance": "group_permutation_importance.csv",
            "test_brightness_contrib_residuals": "test_brightness_contrib_residuals.csv",
            "examples_worst_abs_residuals": "examples_worst_abs_residuals.csv",
        },
    }
    with open(os.path.join(cfg.output_dir, "explain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5b explainability for fusion MLP")
    p.add_argument(
        "--step5_run_summary",
        type=str,
        default=os.path.join("final_project_artifacts", "step5_fusion_mlp", "run_summary.json"),
    )
    p.add_argument(
        "--step5_ckpt",
        type=str,
        default=os.path.join("final_project_artifacts", "step5_fusion_mlp", "best_model.pt"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step5b_explain_mlp"),
    )
    p.add_argument("--n_group_permutations", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Step5BConfig(
        step5_run_summary=args.step5_run_summary,
        step5_ckpt=args.step5_ckpt,
        output_dir=args.output_dir,
        seed=args.seed,
        n_group_permutations=args.n_group_permutations,
    )
    summary = run(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

