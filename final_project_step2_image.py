"""
Final Project - Step 2
Image-only regression pipeline using transfer-learning embeddings.

This step uses the same splits from Step 1 and evaluates image-derived
representations for price prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class Step2Config:
    step1_dir: str = os.path.join("final_project_artifacts", "step1_tabular")
    image_dir: str = os.path.join("socal2", "socal_pics")
    output_dir: str = os.path.join("final_project_artifacts", "step2_image")
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42
    use_pretrained: bool = True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class HouseImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, image_size: int):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tfm = T.Compose(
            [
                T.Resize(int(image_size * 1.15)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = int(row["image_id"])
        path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y = float(row["price"])
        return x, y, image_id


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


def filter_existing_images(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    keep = []
    for _, row in df.iterrows():
        image_id = int(row["image_id"])
        keep.append(os.path.exists(os.path.join(image_dir, f"{image_id}.jpg")))
    return df.loc[keep].reset_index(drop=True)


def build_encoder(use_pretrained: bool, device: torch.device) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    base = resnet18(weights=weights)
    # Remove classification layer to obtain 512-d embeddings.
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.eval()
    encoder.to(device)
    return encoder


@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    image_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    encoder: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = HouseImageDataset(df=df, image_dir=image_dir, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_feats: List[np.ndarray] = []
    all_prices: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []

    for x, y, image_id in loader:
        x = x.to(device, non_blocking=True)
        f = encoder(x)  # [B,512,1,1]
        f = f.view(f.shape[0], -1).cpu().numpy()
        all_feats.append(f)
        all_prices.append(y.numpy())
        all_ids.append(image_id.numpy())

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_prices, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def evaluate_predictions(
    model_name: str,
    y_true_price: np.ndarray,
    pred_log_price: np.ndarray,
) -> Dict[str, float]:
    pred_price = np.expm1(pred_log_price)
    pred_price = np.clip(pred_price, a_min=0.0, a_max=None)
    y_true_log = np.log1p(y_true_price)
    return {
        "model": model_name,
        "r2_price": float(r2_score(y_true_price, pred_price)),
        "mae_price": float(mean_absolute_error(y_true_price, pred_price)),
        "rmse_price": _rmse(y_true_price, pred_price),
        "r2_log_price": float(r2_score(y_true_log, pred_log_price)),
        "mae_log_price": float(mean_absolute_error(y_true_log, pred_log_price)),
        "rmse_log_price": _rmse(y_true_log, pred_log_price),
    }


def run(cfg: Step2Config) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)
    emb_dir = os.path.join(cfg.output_dir, "embeddings")
    _ensure_dir(emb_dir)

    train_df, val_df, test_df = load_splits(cfg.step1_dir)
    train_df = filter_existing_images(train_df, cfg.image_dir)
    val_df = filter_existing_images(val_df, cfg.image_dir)
    test_df = filter_existing_images(test_df, cfg.image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(use_pretrained=cfg.use_pretrained, device=device)

    split_map = {"train": train_df, "val": val_df, "test": test_df}
    feats: Dict[str, np.ndarray] = {}
    prices: Dict[str, np.ndarray] = {}

    for split_name, df in split_map.items():
        out_npz = os.path.join(emb_dir, f"{split_name}_resnet18_embeddings.npz")
        if os.path.exists(out_npz):
            obj = np.load(out_npz)
            feats[split_name] = obj["X"]
            prices[split_name] = obj["y"]
            continue

        X, y, image_ids = extract_embeddings(
            df=df,
            image_dir=cfg.image_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            encoder=encoder,
            device=device,
        )
        np.savez_compressed(out_npz, X=X, y=y, image_id=image_ids)
        feats[split_name] = X
        prices[split_name] = y

    X_train = feats["train"]
    X_val = feats["val"]
    X_test = feats["test"]
    y_train = prices["train"]
    y_val = prices["val"]
    y_test = prices["test"]

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    pca = PCA(n_components=64, random_state=cfg.seed)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    models = {
        "img_linear_raw512": LinearRegression(),
        "img_ridge_raw512": Ridge(alpha=1.0, random_state=cfg.seed),
        "img_ridge_pca64": Ridge(alpha=1.0, random_state=cfg.seed),
        "img_gradient_boosting_pca64": GradientBoostingRegressor(
            random_state=cfg.seed,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        ),
    }

    val_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []
    fitted_models: Dict[str, object] = {}

    for name, model in models.items():
        if name.endswith("pca64"):
            model.fit(X_train_pca, y_train_log)
            val_pred_log = model.predict(X_val_pca)
            test_pred_log = model.predict(X_test_pca)
        else:
            model.fit(X_train, y_train_log)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)

        val_rows.append(evaluate_predictions(name, y_val, val_pred_log))
        test_rows.append(evaluate_predictions(name, y_test, test_pred_log))
        fitted_models[name] = model

    # Naive mean baseline in log-space for transparency.
    mean_log = float(np.mean(y_train_log))
    val_rows.append(
        evaluate_predictions(
            "img_naive_mean_log_price",
            y_val,
            np.full_like(y_val_log, fill_value=mean_log, dtype=float),
        )
    )
    test_rows.append(
        evaluate_predictions(
            "img_naive_mean_log_price",
            y_test,
            np.full_like(y_test_log, fill_value=mean_log, dtype=float),
        )
    )

    val_metrics_df = pd.DataFrame(val_rows).sort_values("r2_log_price", ascending=False)
    test_metrics_df = pd.DataFrame(test_rows).sort_values("r2_log_price", ascending=False)
    val_metrics_df.to_csv(os.path.join(cfg.output_dir, "validation_metrics.csv"), index=False)
    test_metrics_df.to_csv(os.path.join(cfg.output_dir, "test_metrics.csv"), index=False)

    best_model_name = str(val_metrics_df.iloc[0]["model"])

    # Refit best model on train+val and evaluate holdout test.
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    X_trainval_pca = pca.fit_transform(X_trainval)
    y_trainval_log = np.log1p(np.concatenate([y_train, y_val], axis=0))

    if best_model_name == "img_linear_raw512":
        best_model = LinearRegression().fit(X_trainval, y_trainval_log)
        holdout_pred_log = best_model.predict(X_test)
    elif best_model_name == "img_ridge_raw512":
        best_model = Ridge(alpha=1.0, random_state=cfg.seed).fit(X_trainval, y_trainval_log)
        holdout_pred_log = best_model.predict(X_test)
    elif best_model_name == "img_ridge_pca64":
        best_model = Ridge(alpha=1.0, random_state=cfg.seed).fit(X_trainval_pca, y_trainval_log)
        holdout_pred_log = best_model.predict(pca.transform(X_test))
    elif best_model_name == "img_gradient_boosting_pca64":
        best_model = GradientBoostingRegressor(
            random_state=cfg.seed,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        ).fit(X_trainval_pca, y_trainval_log)
        holdout_pred_log = best_model.predict(pca.transform(X_test))
    else:
        holdout_pred_log = np.full_like(y_test_log, fill_value=float(np.mean(y_trainval_log)), dtype=float)

    holdout = evaluate_predictions("holdout_selected_model", y_test, holdout_pred_log)
    run_summary = {
        "config": asdict(cfg),
        "split_sizes_after_image_filter": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "embedding_dim": int(X_train.shape[1]),
        "best_validation_model": best_model_name,
        "final_holdout_metrics": holdout,
        "device": device.type,
    }
    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--step1_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step1_tabular"),
    )
    p.add_argument("--image_dir", type=str, default=os.path.join("socal2", "socal_pics"))
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("final_project_artifacts", "step2_image"),
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scratch_encoder", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Step2Config(
        step1_dir=args.step1_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_pretrained=(not bool(args.scratch_encoder)),
    )
    summary = run(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

