"""
Final Project - Step 3
Multimodal fusion: tabular features (Step 1 splits) + image embeddings (Step 2).

Modular knobs (CLI) so you can swap tabular feature sets, image representations,
and regressors without rewriting the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from final_project_step1_tabular import Step1Config, build_preprocessor, get_feature_sets


@dataclass(frozen=True)
class Step3Config:
    step1_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    embeddings_dir: str = os.path.join("final_project_artifacts", "step2_image", "embeddings")
    output_dir: str = os.path.join("final_project_artifacts", "step3_fusion")
    seed: int = 42
    # Tabular: subset of keys from get_feature_sets (e.g. core_beds_baths_sqft, core_plus_city)
    tabular_feature_sets: Tuple[str, ...] = ("core_beds_baths_sqft", "core_plus_city")
    # Image: "raw" (scaled 512-d), "pca64"
    image_reprs: Tuple[str, ...] = ("raw", "pca64")
    # Regressor names; see MODEL_BUILDERS
    # Default excludes random forest: on wide fused one-hot + embeddings it is slow on CPU.
    # Pass `--models ...` to add `random_forest` or other keys from build_model_factories.
    model_names: Tuple[str, ...] = ("linear_regression", "ridge", "gradient_boosting")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _to_dense_2d(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def load_splits(step1_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(step1_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(step1_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(step1_dir, "test_split.csv"))
    for df in (train_df, val_df, test_df):
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df.dropna(subset=["image_id", "price"], inplace=True)
        df["image_id"] = df["image_id"].astype(int)
        df["price"] = df["price"].astype(float)
    return train_df, val_df, test_df


def load_embeddings_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["X"].astype(np.float32), z["image_id"].astype(np.int64)


def merge_tabular_with_embeddings(
    tab_df: pd.DataFrame,
    X_emb: np.ndarray,
    emb_image_ids: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Align rows of tab_df with embedding rows by image_id (inner join)."""
    idx_map = pd.DataFrame({"image_id": emb_image_ids, "_emb_row": np.arange(len(emb_image_ids), dtype=np.int64)})
    merged = tab_df.merge(idx_map, on="image_id", how="inner", validate="many_to_one")
    X_aligned = X_emb[merged["_emb_row"].values]
    merged = merged.drop(columns=["_emb_row"])
    return merged.reset_index(drop=True), X_aligned


def transform_tabular_block(
    preprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tabular_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tr = train_df[tabular_cols].copy()
    X_va = val_df[tabular_cols].copy()
    X_te = test_df[tabular_cols].copy()
    T_tr = _to_dense_2d(preprocessor.fit_transform(X_tr))
    T_va = _to_dense_2d(preprocessor.transform(X_va))
    T_te = _to_dense_2d(preprocessor.transform(X_te))
    return T_tr, T_va, T_te


def build_image_block(
    repr_name: str,
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {"repr": repr_name}
    if repr_name == "raw":
        scaler = StandardScaler()
        I_tr = scaler.fit_transform(X_tr)
        I_va = scaler.transform(X_va)
        I_te = scaler.transform(X_te)
        meta["scaler_mean_shape"] = list(scaler.mean_.shape)
        return I_tr, I_va, I_te, meta
    if repr_name == "pca64":
        pca = PCA(n_components=64, random_state=seed)
        I_tr = pca.fit_transform(X_tr)
        I_va = pca.transform(X_va)
        I_te = pca.transform(X_te)
        meta["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        return I_tr, I_va, I_te, meta
    raise ValueError(f"Unknown image_repr: {repr_name}. Use: raw, pca64")


def hstack_tabular_image(T: np.ndarray, I: np.ndarray) -> np.ndarray:
    return np.hstack([T, I.astype(np.float64)])


def metrics_row(
    experiment_id: str,
    tabular_fs: str,
    image_repr: str,
    model_name: str,
    y_true_price: np.ndarray,
    pred_log: np.ndarray,
) -> Dict[str, object]:
    pred_price = np.expm1(pred_log)
    pred_price = np.clip(pred_price, a_min=0.0, a_max=None)
    y_log = np.log1p(y_true_price)
    return {
        "experiment_id": experiment_id,
        "tabular_feature_set": tabular_fs,
        "image_repr": image_repr,
        "model": model_name,
        "r2_price": float(r2_score(y_true_price, pred_price)),
        "mae_price": float(mean_absolute_error(y_true_price, pred_price)),
        "rmse_price": _rmse(y_true_price, pred_price),
        "r2_log_price": float(r2_score(y_log, pred_log)),
        "mae_log_price": float(mean_absolute_error(y_log, pred_log)),
        "rmse_log_price": _rmse(y_log, pred_log),
    }


def build_model_factories(seed: int) -> Dict[str, Callable[[], object]]:
    # Tree models use settings tuned for wide fused features (one-hot city + image block).
    # `gradient_boosting` uses histogram GB for speed at this scale vs classic sklearn GBRT.
    return {
        "linear_regression": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0, random_state=seed),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=140,
            learning_rate=0.08,
            max_depth=6,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
    }


def run(cfg: Step3Config) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)
    s1_cfg = Step1Config(seed=cfg.seed)
    feature_sets = get_feature_sets(s1_cfg)

    for name in cfg.tabular_feature_sets:
        if name not in feature_sets:
            raise ValueError(
                f"Unknown tabular_feature_set '{name}'. "
                f"Valid: {sorted(feature_sets.keys())}"
            )

    train_df, val_df, test_df = load_splits(cfg.step1_dir)
    emb_train = os.path.join(cfg.embeddings_dir, "train_resnet18_embeddings.npz")
    emb_val = os.path.join(cfg.embeddings_dir, "val_resnet18_embeddings.npz")
    emb_test = os.path.join(cfg.embeddings_dir, "test_resnet18_embeddings.npz")
    for p in (emb_train, emb_val, emb_test):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing embedding file: {p}. Run final_project_step2_image.py first.")

    Xtr_e, idtr = load_embeddings_npz(emb_train)
    Xva_e, idva = load_embeddings_npz(emb_val)
    Xte_e, idte = load_embeddings_npz(emb_test)

    train_df, Xtr_e = merge_tabular_with_embeddings(train_df, Xtr_e, idtr)
    val_df, Xva_e = merge_tabular_with_embeddings(val_df, Xva_e, idva)
    test_df, Xte_e = merge_tabular_with_embeddings(test_df, Xte_e, idte)

    y_train = train_df["price"].to_numpy(dtype=np.float64)
    y_val = val_df["price"].to_numpy(dtype=np.float64)
    y_test = test_df["price"].to_numpy(dtype=np.float64)
    y_train_log = np.log1p(y_train)

    factories = build_model_factories(cfg.seed)
    for m in cfg.model_names:
        if m not in factories:
            raise ValueError(f"Unknown model '{m}'. Valid: {sorted(factories.keys())}")

    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for fs_name in cfg.tabular_feature_sets:
        fs_cols = feature_sets[fs_name]
        tabular_cols = [*fs_cols["numeric"], *fs_cols["categorical"]]
        pre = build_preprocessor(
            numeric_cols=list(fs_cols["numeric"]),
            categorical_cols=list(fs_cols["categorical"]),
        )
        T_tr, T_va, T_te = transform_tabular_block(pre, train_df, val_df, test_df, tabular_cols)

        for img_repr in cfg.image_reprs:
            I_tr, I_va, I_te, _img_meta = build_image_block(img_repr, Xtr_e, Xva_e, Xte_e, cfg.seed)
            F_tr = hstack_tabular_image(T_tr, I_tr)
            F_va = hstack_tabular_image(T_va, I_va)
            F_te = hstack_tabular_image(T_te, I_te)

            for model_name in cfg.model_names:
                model = factories[model_name]()
                model.fit(F_tr, y_train_log)
                pred_va = model.predict(F_va)
                pred_te = model.predict(F_te)
                eid = f"{fs_name}__{img_repr}__{model_name}"
                val_rows.append(metrics_row(eid, fs_name, img_repr, model_name, y_val, pred_va))
                test_rows.append(metrics_row(eid, fs_name, img_repr, model_name, y_test, pred_te))
                print(f"[fusion] done {eid}", flush=True)

    metrics_val_df = pd.DataFrame(val_rows).sort_values("r2_log_price", ascending=False)
    metrics_test_df = pd.DataFrame(test_rows).sort_values("r2_log_price", ascending=False)
    metrics_val_df.to_csv(os.path.join(cfg.output_dir, "validation_metrics.csv"), index=False)
    metrics_test_df.to_csv(os.path.join(cfg.output_dir, "test_metrics.csv"), index=False)

    best = metrics_val_df.iloc[0]
    best_eid = str(best["experiment_id"])
    best_fs = str(best["tabular_feature_set"])
    best_img = str(best["image_repr"])
    best_model_name = str(best["model"])

    # Refit best config on train+val and evaluate holdout test.
    fs_cols = feature_sets[best_fs]
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

    I_trv, I_te, _ = _build_image_block_train_test_only(best_img, X_trainval_e, Xte_e, cfg.seed)

    F_trv = hstack_tabular_image(T_trv, I_trv)
    F_te = hstack_tabular_image(T_te, I_te)
    best_est = factories[best_model_name]()
    best_est.fit(F_trv, y_trainval_log)
    pred_te = best_est.predict(F_te)
    holdout = metrics_row("holdout_best_from_val", best_fs, best_img, best_model_name, y_test, pred_te)

    ablation = {
        "config": asdict(cfg),
        "split_sizes_after_merge": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "best_from_validation": {
            "experiment_id": best_eid,
            "tabular_feature_set": best_fs,
            "image_repr": best_img,
            "model": best_model_name,
        },
        "holdout_test": holdout,
        "top_5_validation_by_r2_log_price": metrics_val_df.head(5).to_dict(orient="records"),
    }
    with open(os.path.join(cfg.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "best_from_validation": ablation["best_from_validation"],
                "holdout_test": ablation["holdout_test"],
            },
            f,
            indent=2,
        )

    return ablation


def _build_image_block_train_test_only(
    repr_name: str,
    X_trainval: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fit image transform on train+val only; transform test."""
    meta: Dict[str, object] = {"repr": repr_name}
    if repr_name == "raw":
        scaler = StandardScaler()
        I_trv = scaler.fit_transform(X_trainval)
        I_te = scaler.transform(X_test)
        return I_trv, I_te, meta
    if repr_name == "pca64":
        pca = PCA(n_components=64, random_state=seed)
        I_trv = pca.fit_transform(X_trainval)
        I_te = pca.transform(X_test)
        meta["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        return I_trv, I_te, meta
    raise ValueError(repr_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: tabular + image fusion (modular)")
    p.add_argument(
        "--step1_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step1_tabular"),
    )
    p.add_argument(
        "--embeddings_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step2_image", "embeddings"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step3_fusion"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tabular_feature_sets",
        type=str,
        default="core_beds_baths_sqft,core_plus_city",
        help="Comma-separated: core_beds_baths_sqft, core_plus_city",
    )
    p.add_argument(
        "--image_reprs",
        type=str,
        default="raw,pca64",
        help="Comma-separated: raw, pca64",
    )
    p.add_argument(
        "--models",
        type=str,
        default="linear_regression,ridge,gradient_boosting",
        help="Comma-separated: linear_regression,ridge,random_forest,gradient_boosting",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tab_sets = tuple(s.strip() for s in args.tabular_feature_sets.split(",") if s.strip())
    img_reprs = tuple(s.strip() for s in args.image_reprs.split(",") if s.strip())
    model_names = tuple(s.strip() for s in args.models.split(",") if s.strip())
    cfg = Step3Config(
        step1_dir=args.step1_dir,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        tabular_feature_sets=tab_sets,
        image_reprs=img_reprs,
        model_names=model_names,
    )
    summary = run(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
