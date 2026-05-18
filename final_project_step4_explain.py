"""
Final Project - Step 4
Explainability for the fused tabular + image model:
  - Grouped permutation importance (numeric vs city one-hot vs image block)
  - Ablations that isolate what vision contributes (zeros / shuffle image block)
  - Residual analysis (where the model is strong vs brittle)
  - Linear-model coefficient snapshot (when the chosen model is linear / ridge)
  - Short, structured "next improvements" notes for the report

Reads best settings from `final_project_artifacts/step3_fusion/run_summary.json`
unless overridden via CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

from final_project_step1_tabular import Step1Config, build_preprocessor, get_feature_sets
from final_project_step3_fusion import (
    build_model_factories,
    hstack_tabular_image,
    load_embeddings_npz,
    load_splits,
    merge_tabular_with_embeddings,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_dense_2d(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _build_image_block_train_test_only(
    repr_name: str,
    X_trainval: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if repr_name == "raw":
        scaler = StandardScaler()
        return scaler.fit_transform(X_trainval), scaler.transform(X_test)
    if repr_name == "pca64":
        pca = PCA(n_components=64, random_state=seed)
        return pca.fit_transform(X_trainval), pca.transform(X_test)
    raise ValueError(f"Unknown image_repr: {repr_name}")


def fused_feature_names(pre, n_img: int, img_tag: str) -> np.ndarray:
    tab = np.asarray(pre.get_feature_names_out(), dtype=str)
    img = np.array([f"{img_tag}_{i}" for i in range(n_img)], dtype=str)
    return np.concatenate([tab, img])


def classify_feature_group(name: str) -> str:
    if name.startswith("num__"):
        return "tabular_numeric"
    if name.startswith("cat__"):
        return "tabular_city_onehot"
    if name.startswith("img_"):
        return "image_block"
    return "other"


def r2_log_price_scorer():
    """Higher is better; y_true is raw price, y_pred is model output in log1p(price) space."""
    return make_scorer(
        lambda y_true, y_pred: float(
            r2_score(np.log1p(np.asarray(y_true, dtype=float)), np.asarray(y_pred, dtype=float))
        ),
        greater_is_better=True,
    )


def load_best_from_run_summary(path: str) -> Tuple[str, str, str]:
    with open(path) as f:
        s = json.load(f)
    b = s["best_from_validation"]
    return str(b["tabular_feature_set"]), str(b["image_repr"]), str(b["model"])


@dataclass(frozen=True)
class ExplainConfig:
    step1_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    embeddings_dir: str = os.path.join("final_project_artifacts", "step2_image", "embeddings")
    step3_run_summary: str = os.path.join("final_project_artifacts", "step3_fusion", "run_summary.json")
    output_dir: str = os.path.join("final_project_artifacts", "step4_explain")
    seed: int = 42
    # Subsample test rows for permutation importance (full test is fine but slower)
    max_perm_samples: int = 2000
    n_perm_repeats: int = 8
    image_dir: str = os.path.join("socal2", "socal_pics")


def build_trainval_test_fused(
    tabular_fs: str,
    image_repr: str,
    seed: int,
    step1_dir: str,
    embeddings_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str], object]:
    s1_cfg = Step1Config(seed=seed)
    feature_sets = get_feature_sets(s1_cfg)
    if tabular_fs not in feature_sets:
        raise ValueError(f"Unknown tabular_fs {tabular_fs}")

    train_df, val_df, test_df = load_splits(step1_dir)
    Xtr_e, idtr = load_embeddings_npz(os.path.join(embeddings_dir, "train_resnet18_embeddings.npz"))
    Xva_e, idva = load_embeddings_npz(os.path.join(embeddings_dir, "val_resnet18_embeddings.npz"))
    Xte_e, idte = load_embeddings_npz(os.path.join(embeddings_dir, "test_resnet18_embeddings.npz"))

    train_df, Xtr_e = merge_tabular_with_embeddings(train_df, Xtr_e, idtr)
    val_df, Xva_e = merge_tabular_with_embeddings(val_df, Xva_e, idva)
    test_df, Xte_e = merge_tabular_with_embeddings(test_df, Xte_e, idte)

    fs_cols = feature_sets[tabular_fs]
    tabular_cols = [*fs_cols["numeric"], *fs_cols["categorical"]]
    pre = build_preprocessor(
        numeric_cols=list(fs_cols["numeric"]),
        categorical_cols=list(fs_cols["categorical"]),
    )

    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    X_trainval_e = np.vstack([Xtr_e, Xva_e])
    y_trainval = trainval_df["price"].to_numpy(dtype=np.float64)
    y_trainval_log = np.log1p(y_trainval)

    X_tr_tab = trainval_df[tabular_cols].copy()
    X_te_tab = test_df[tabular_cols].copy()
    T_trv = _to_dense_2d(pre.fit_transform(X_tr_tab))
    T_te = _to_dense_2d(pre.transform(X_te_tab))

    I_trv, I_te = _build_image_block_train_test_only(image_repr, X_trainval_e, Xte_e, seed)
    F_trv = hstack_tabular_image(T_trv, I_trv)
    F_te = hstack_tabular_image(T_te, I_te)
    y_test = test_df["price"].to_numpy(dtype=np.float64)

    return F_trv, F_te, T_trv, T_te, I_te, y_trainval_log, y_test, test_df, tabular_cols, pre


def run(cfg: ExplainConfig, tabular_fs: str, image_repr: str, model_name: str) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)

    factories = build_model_factories(cfg.seed)
    if model_name not in factories:
        raise ValueError(f"Unknown model {model_name}")

    F_trv, F_te, T_trv, T_te, I_te, y_trv_log, y_test, test_df, tabular_cols, pre = build_trainval_test_fused(
        tabular_fs=tabular_fs,
        image_repr=image_repr,
        seed=cfg.seed,
        step1_dir=cfg.step1_dir,
        embeddings_dir=cfg.embeddings_dir,
    )

    model = factories[model_name]()
    model.fit(F_trv, y_trv_log)
    pred_log = model.predict(F_te)
    pred_price = np.clip(np.expm1(pred_log), 0.0, None)

    n_img = int(I_te.shape[1])
    img_tag = "img_raw" if image_repr == "raw" else "img_pca"
    names = fused_feature_names(pre, n_img, img_tag)

    # --- Permutation importance (per-feature, then aggregate by group)
    rng = np.random.default_rng(cfg.seed)
    n_use = min(cfg.max_perm_samples, F_te.shape[0])
    idx = rng.choice(F_te.shape[0], size=n_use, replace=False)
    Xp = F_te[idx]
    yp = y_test[idx]

    perm = permutation_importance(
        model,
        Xp,
        yp,
        n_repeats=cfg.n_perm_repeats,
        random_state=cfg.seed,
        n_jobs=-1,
        scoring=r2_log_price_scorer(),
    )
    imp_mean = perm.importances_mean
    imp_std = perm.importances_std

    perm_df = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": imp_mean,
            "importance_std": imp_std,
            "group": [classify_feature_group(str(n)) for n in names],
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(os.path.join(cfg.output_dir, "permutation_importance_by_feature.csv"), index=False)

    group_summary = (
        perm_df.groupby("group", as_index=False)
        .agg(sum_importance=("importance_mean", "sum"), mean_importance=("importance_mean", "mean"), n=("feature", "count"))
        .sort_values("sum_importance", ascending=False)
    )
    group_summary.to_csv(os.path.join(cfg.output_dir, "permutation_importance_by_group.csv"), index=False)

    # --- Ablations on full test (same fitted fused model)
    y_test_log = np.log1p(y_test)
    r2_log_full = float(r2_score(y_test_log, pred_log))

    # Tabular-only linear baseline (same train+val log target, same T block)
    tab_only = LinearRegression()
    tab_only.fit(T_trv, y_trv_log)
    pred_tab_log = tab_only.predict(T_te)
    r2_log_tab_only = float(r2_score(y_test_log, pred_tab_log))

    I_zero = np.zeros_like(I_te)
    pred_zero_img = model.predict(np.hstack([T_te, I_zero]))
    r2_log_zero_img = float(r2_score(y_test_log, pred_zero_img))

    shuf_idx = rng.permutation(len(I_te))
    pred_shuf_img = model.predict(np.hstack([T_te, I_te[shuf_idx]]))
    r2_log_shuf_img = float(r2_score(y_test_log, pred_shuf_img))

    ablation = {
        "r2_log_price_fused_model": r2_log_full,
        "r2_log_price_tabular_only_linear_on_T": r2_log_tab_only,
        "r2_log_price_fused_but_image_zeroed": r2_log_zero_img,
        "r2_log_price_fused_but_image_shuffled": r2_log_shuf_img,
        "interpretation": {
            "tabular_only_vs_fused_delta_r2_log": r2_log_full - r2_log_tab_only,
            "zero_image_vs_fused_delta_r2_log": r2_log_zero_img - r2_log_full,
            "shuffle_image_vs_fused_delta_r2_log": r2_log_shuf_img - r2_log_full,
        },
    }
    with open(os.path.join(cfg.output_dir, "ablation_image_contribution.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    # --- Residuals
    res_log = y_test_log - pred_log
    res_price = y_test - pred_price
    out_df = test_df[["image_id", "price"]].copy().reset_index(drop=True)
    out_df["pred_log_price"] = pred_log
    out_df["pred_price"] = pred_price
    out_df["residual_log"] = res_log
    out_df["residual_price"] = res_price
    out_df["abs_residual_price"] = np.abs(res_price)
    out_df.to_csv(os.path.join(cfg.output_dir, "test_predictions_and_residuals.csv"), index=False)

    worst_over = out_df.nlargest(25, "residual_price")  # predicted too low
    worst_under = out_df.nsmallest(25, "residual_price")  # predicted too high
    worst_over.to_csv(os.path.join(cfg.output_dir, "examples_largest_positive_residual.csv"), index=False)
    worst_under.to_csv(os.path.join(cfg.output_dir, "examples_largest_negative_residual.csv"), index=False)

    # Scatter: predicted vs residual (log)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(pred_log, res_log, s=8, alpha=0.35, c="#4C72B0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Predicted log1p(price)")
    ax.set_ylabel("Residual (true - pred) in log space")
    ax.set_title("Test residuals vs fused model prediction")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "residuals_vs_pred_log.png"), dpi=150)
    plt.close(fig)

    # --- Linear coefficients (if available)
    coef_payload: Dict[str, object] = {"available": False}
    if hasattr(model, "coef_") and model.coef_ is not None and model.coef_.ndim == 1:
        coef = np.asarray(model.coef_).ravel()
        order = np.argsort(np.abs(coef))[-30:]
        coef_df = pd.DataFrame({"feature": names[order], "coefficient": coef[order]})
        coef_df.to_csv(os.path.join(cfg.output_dir, "linear_top30_coefficients.csv"), index=False)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.barh(coef_df["feature"], coef_df["coefficient"], color="#55A868")
        ax2.set_title(f"Top coefficients by magnitude ({model_name}, log target)")
        ax2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(cfg.output_dir, "linear_top30_coefficients.png"), dpi=150)
        plt.close(fig2)
        img_mask = np.array([str(n).startswith("img_") for n in names], dtype=bool)
        img_coef_df = pd.DataFrame({"feature": names[img_mask], "coefficient": coef[img_mask]})
        img_by_abs = img_coef_df.reindex(img_coef_df["coefficient"].abs().sort_values(ascending=False).index)
        coef_payload = {
            "available": True,
            "top5_image_dims_by_abs_coefficient": img_by_abs.head(5).to_dict(orient="records"),
            "top5_positive_image_dims": img_coef_df.nlargest(5, "coefficient").to_dict(orient="records"),
            "top5_negative_image_dims": img_coef_df.nsmallest(5, "coefficient").to_dict(orient="records"),
        }

    # --- Partial dependence on standardized num__sqft in fused feature space
    pdp_note = ""
    try:
        sqft_name = "num__sqft"
        if sqft_name in names:
            j = int(np.where(names == sqft_name)[0][0])
            disp = PartialDependenceDisplay.from_estimator(
                model,
                F_te,
                features=[j],
                kind="average",
                grid_resolution=25,
            )
            disp.figure_.savefig(os.path.join(cfg.output_dir, "pdp_num_sqft.png"), dpi=150, bbox_inches="tight")
            plt.close("all")
            pdp_note = "PDP for standardized num__sqft in fused feature space (average effect)."
        else:
            pdp_note = "num__sqft not found in feature names; skip PDP."
    except Exception as e:  # noqa: BLE001
        pdp_note = f"PDP skipped: {e}"

    improvements = {
        "representation": [
            "Replace frozen ResNet18 pooled vector with a stronger backbone (ViT-B/16, CLIP image encoder) or multi-crop / higher resolution.",
            "Add learned fusion: small MLP on [tabular; emb] with dropout and weight decay instead of purely linear stacking.",
            "Use patch-level tokens + attention pooling rather than a single global vector.",
        ],
        "vision_supervision_and_domain_gap": [
            "Light fine-tuning of the image trunk on a proxy task (e.g., hedonic attributes) or contrastive learning aligned to price bins before regression.",
            "Domain adaptation: CMP-facade segmentation (Assignment 3) can yield ratio features (vegetation, window area) transferable if fine-tuned on a small labeled SoCal subset.",
        ],
        "tabular_and_leakage_control": [
            "Audit whether `citi` proxies neighborhood wealth; try ablations without city to see if images pick up street/neighborhood cues (fairness slide).",
            "Add lat-long binning or census tract controls deliberately *not* from pixels to separate structure vs location effects.",
        ],
        "explainability_go_deeper": [
            "Integrated gradients / Grad-CAM on a fine-tuned CNN trunk for individual high-residual listings.",
            "SHAP KernelExplainer on a small test subset for the fused MLP (expensive but presentation-friendly at n<500).",
            "Error cohort analysis: bucket by property age proxy, lot size if available, or price decile — report where image ablation hurts most.",
        ],
        "metrics": [
            "Report multiple metrics: MAE in $ on log-scale, Gini-style decile calibration, and coverage of prediction intervals (e.g., conformal after fusion).",
        ],
    }

    summary = {
        "config": {
            "tabular_feature_set": tabular_fs,
            "image_repr": image_repr,
            "model": model_name,
            "step1_dir": cfg.step1_dir,
            "embeddings_dir": cfg.embeddings_dir,
            "max_perm_samples": cfg.max_perm_samples,
            "n_perm_repeats": cfg.n_perm_repeats,
        },
        "test_metrics_fused": {
            "r2_log_price": r2_log_full,
            "r2_price": float(r2_score(y_test, pred_price)),
            "mae_price": float(mean_absolute_error(y_test, pred_price)),
        },
        "ablation_image_contribution": ablation,
        "permutation_group_summary_csv": "permutation_importance_by_group.csv",
        "coefficients": coef_payload,
        "pdp_note": pdp_note,
        "improvement_ideas": improvements,
    }
    with open(os.path.join(cfg.output_dir, "explain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: explainability for fused model")
    p.add_argument(
        "--step3_run_summary",
        type=str,
        default=os.path.join("final_project_artifacts", "step3_fusion", "run_summary.json"),
    )
    p.add_argument("--tabular_feature_set", type=str, default=None)
    p.add_argument("--image_repr", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step4_explain"),
    )
    p.add_argument("--max_perm_samples", type=int, default=2000)
    p.add_argument("--n_perm_repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tabular_feature_set and args.image_repr and args.model:
        tab_fs, img_r, mod = args.tabular_feature_set, args.image_repr, args.model
    else:
        tab_fs, img_r, mod = load_best_from_run_summary(args.step3_run_summary)

    cfg = ExplainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        max_perm_samples=args.max_perm_samples,
        n_perm_repeats=args.n_perm_repeats,
    )
    summary = run(cfg, tabular_fs=tab_fs, image_repr=img_r, model_name=mod)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
