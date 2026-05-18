"""
Final Project - Step 1
Reproducible EDA + metadata-only price regression baselines.

Goals for this step:
1) Load and validate the SoCal housing metadata CSV.
2) Run EDA summaries used in the report/presentation.
3) Train metadata-only baselines to establish a non-vision floor.
4) Save artifacts (tables, figures, metrics) for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Step1Config:
    csv_path: str = "socal2.csv"
    output_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    seed: int = 42
    test_frac: float = 0.15
    val_frac: float = 0.15  # fraction of the total dataset
    target_col: str = "price"
    numeric_cols: Tuple[str, ...] = ("bed", "bath", "sqft")
    categorical_cols: Tuple[str, ...] = ("citi",)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_and_clean(cfg: Step1Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    needed = [cfg.target_col, *cfg.numeric_cols, *cfg.categorical_cols, "image_id"]
    present = [c for c in needed if c in df.columns]
    df = df[present].copy()

    for col in [cfg.target_col, *cfg.numeric_cols]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only plausible positive-valued homes.
    df = df.dropna(subset=[cfg.target_col]).copy()
    df = df[df[cfg.target_col] > 0].copy()
    if "sqft" in df.columns:
        df = df[df["sqft"] > 0].copy()

    return df.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_frac <= 0 or val_frac <= 0 or (test_frac + val_frac) >= 1.0:
        raise ValueError("test_frac and val_frac must be > 0 and sum to < 1.")

    trainval_df, test_df = train_test_split(
        df, test_size=test_frac, random_state=seed
    )

    val_rel = val_frac / (1.0 - test_frac)
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_rel, random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
) -> Dict[str, float]:
    estimator.fit(X_train, y_train_log)
    pred_log = estimator.predict(X_eval)
    pred_price = np.expm1(pred_log)

    return {
        "model": model_name,
        "r2_price": float(r2_score(y_eval, pred_price)),
        "mae_price": float(mean_absolute_error(y_eval, pred_price)),
        "rmse_price": _rmse(y_eval, pred_price),
        "r2_log_price": float(r2_score(np.log1p(y_eval), pred_log)),
        "mae_log_price": float(mean_absolute_error(np.log1p(y_eval), pred_log)),
        "rmse_log_price": _rmse(np.log1p(y_eval), pred_log),
    }


def build_models(preprocessor: ColumnTransformer, seed: int) -> List[Tuple[str, Pipeline]]:
    return [
        (
            "linear_regression",
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        (
            "ridge",
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    ("model", Ridge(alpha=1.0, random_state=seed)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        random_state=seed,
                        n_jobs=-1,
                    )),
                ]
            ),
        ),
        (
            "gradient_boosting",
            Pipeline(
                steps=[
                    ("pre", preprocessor),
                    ("model", GradientBoostingRegressor(
                        random_state=seed,
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.9,
                    )),
                ]
            ),
        ),
    ]


def get_feature_sets(cfg: Step1Config) -> Dict[str, Dict[str, List[str]]]:
    return {
        "core_beds_baths_sqft": {
            "numeric": list(cfg.numeric_cols),
            "categorical": [],
        },
        "core_plus_city": {
            "numeric": list(cfg.numeric_cols),
            "categorical": [c for c in cfg.categorical_cols if c in ("citi", "n_citi")],
        },
    }


def save_eda(df: pd.DataFrame, cfg: Step1Config, output_dir: str) -> Dict[str, object]:
    price = df[cfg.target_col].astype(float)
    cols = [cfg.target_col, *cfg.numeric_cols]
    corr = df[cols].corr(numeric_only=True)

    summary_table = pd.DataFrame(
        {
            "mean": df[cols].mean(numeric_only=True),
            "std": df[cols].std(numeric_only=True),
            "min": df[cols].min(numeric_only=True),
            "p25": df[cols].quantile(0.25, numeric_only=True),
            "median": df[cols].median(numeric_only=True),
            "p75": df[cols].quantile(0.75, numeric_only=True),
            "max": df[cols].max(numeric_only=True),
        }
    )
    summary_table.to_csv(os.path.join(output_dir, "summary_stats.csv"))
    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(price, bins=60, color="#4C72B0", alpha=0.9)
    plt.title("Price Distribution")
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "price_hist.png"), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(7, 4))
    plt.hist(np.log1p(price), bins=60, color="#55A868", alpha=0.9)
    plt.title("log1p(Price) Distribution")
    plt.xlabel("log1p(price)")
    plt.ylabel("Count")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "log_price_hist.png"), dpi=150)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig3)

    return {
        "rows": int(len(df)),
        "price_mean": float(price.mean()),
        "price_median": float(price.median()),
        "price_std": float(price.std()),
        "price_min": float(price.min()),
        "price_max": float(price.max()),
    }


def run(cfg: Step1Config) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)
    df = load_and_clean(cfg)
    train_df, val_df, test_df = make_splits(
        df=df,
        test_frac=cfg.test_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    train_df.to_csv(os.path.join(cfg.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.output_dir, "test_split.csv"), index=False)

    eda_summary = save_eda(df=df, cfg=cfg, output_dir=cfg.output_dir)

    y_train = train_df[cfg.target_col].astype(float).to_numpy()
    y_train_log = np.log1p(y_train)
    y_val = val_df[cfg.target_col].astype(float).to_numpy()
    y_test = test_df[cfg.target_col].astype(float).to_numpy()

    val_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []
    fitted_models: Dict[str, Pipeline] = {}
    feature_sets = get_feature_sets(cfg)

    for fs_name, fs_cols in feature_sets.items():
        use_cols = [*fs_cols["numeric"], *fs_cols["categorical"]]
        X_train = train_df[use_cols].copy()
        X_val = val_df[use_cols].copy()
        X_test = test_df[use_cols].copy()

        preprocessor = build_preprocessor(
            numeric_cols=fs_cols["numeric"],
            categorical_cols=fs_cols["categorical"],
        )
        models = build_models(preprocessor=preprocessor, seed=cfg.seed)

        for model_name, pipeline in models:
            full_name = f"{fs_name}__{model_name}"
            val_metrics = evaluate_model(
                model_name=full_name,
                estimator=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_eval=X_val,
                y_eval=y_val,
            )
            val_metrics["feature_set"] = fs_name
            val_rows.append(val_metrics)
            fitted_models[full_name] = pipeline

            te = evaluate_model(
                model_name=full_name,
                estimator=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_eval=X_test,
                y_eval=y_test,
            )
            te["feature_set"] = fs_name
            test_rows.append(te)

    val_df_metrics = pd.DataFrame(val_rows).sort_values("r2_log_price", ascending=False)
    val_df_metrics.to_csv(os.path.join(cfg.output_dir, "validation_metrics.csv"), index=False)

    test_df_metrics = pd.DataFrame(test_rows).sort_values("r2_log_price", ascending=False)
    test_df_metrics.to_csv(os.path.join(cfg.output_dir, "test_metrics.csv"), index=False)

    # Select best model from validation scores across feature sets.
    best_model_name = str(val_df_metrics.iloc[0]["model"])
    best_feature_set = str(val_df_metrics.iloc[0]["feature_set"])
    best_cols = feature_sets[best_feature_set]
    X_trainval = pd.concat(
        [
            train_df[[*best_cols["numeric"], *best_cols["categorical"]]],
            val_df[[*best_cols["numeric"], *best_cols["categorical"]]],
        ],
        axis=0,
    ).reset_index(drop=True)
    X_test = test_df[[*best_cols["numeric"], *best_cols["categorical"]]].copy()
    y_trainval_log = np.log1p(
        pd.concat([train_df[cfg.target_col], val_df[cfg.target_col]], axis=0)
        .astype(float)
        .to_numpy()
    )

    best_model = fitted_models[best_model_name]
    best_model.fit(X_trainval, y_trainval_log)
    final_pred = np.expm1(best_model.predict(X_test))
    final_pred = np.clip(final_pred, a_min=0.0, a_max=None)
    final_pred_log = np.log1p(final_pred)

    final_holdout = {
        "selected_model_from_validation": best_model_name,
        "selected_feature_set": best_feature_set,
        "holdout_r2_price": float(r2_score(y_test, final_pred)),
        "holdout_mae_price": float(mean_absolute_error(y_test, final_pred)),
        "holdout_rmse_price": _rmse(y_test, final_pred),
        "holdout_r2_log_price": float(r2_score(np.log1p(y_test), final_pred_log)),
    }

    run_summary = {
        "config": asdict(cfg),
        "eda_summary": eda_summary,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "best_validation_model": best_model_name,
        "final_holdout_metrics": final_holdout,
    }

    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="socal2.csv")
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step1_tabular"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--val_frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Step1Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )
    summary = run(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

