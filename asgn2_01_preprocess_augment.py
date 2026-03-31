"""
Assignment 2 - Step 3 style modular file:
Preprocessing + augmentation pipeline for a CNN/ViT-based price-bracket classification task.

What it does:
1) Loads `socal2.csv`
2) Filters rows whose images exist at `image_dir/{image_id}.jpg`
3) Creates `price_class` labels (quantile/tertile bins)
4) Creates stratified train/val/test splits
5) Saves split CSVs + tabular normalization stats for later steps
6) Defines torchvision transforms (train/val) compatible with both CNNs and ViT
7) Defines a PyTorch Dataset + DataLoader builder (used later for training)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from PIL import Image
except ModuleNotFoundError:
    torch = None
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    T = None  # type: ignore
    Image = None  # type: ignore


DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class PreprocessConfig:
    csv_path: str
    image_dir: str
    output_dir: str

    image_id_col: str = "image_id"
    image_ext: str = ".jpg"

    price_col: str = "price"
    label_col: str = "price_class"
    num_classes: int = 3

    bin_method: str = "quantile"  # "quantile" or "uniform"
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # Tabular normalization stats are computed and saved for later fusion models.
    tabular_cols: Tuple[str, ...] = ("bed", "bath", "sqft", "n_citi")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _image_path(image_dir: str, image_id: int | str, image_ext: str) -> str:
    return os.path.join(image_dir, f"{int(image_id)}{image_ext}")


def load_and_clean_metadata(cfg: PreprocessConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    # Basic numeric coercion (bath/sqft may be floats in the CSV).
    for col in (cfg.price_col, *cfg.tabular_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical fields.
    needed = [cfg.image_id_col, cfg.price_col, *cfg.tabular_cols]
    needed = [c for c in needed if c in df.columns]
    df = df.dropna(subset=needed).copy()

    return df


def filter_rows_with_existing_images(cfg: PreprocessConfig, df: pd.DataFrame) -> pd.DataFrame:
    # Filter to only rows with existing image files.
    keep_mask = []
    missing: List[int] = []
    for _, row in df.iterrows():
        img_id = row[cfg.image_id_col]
        p = _image_path(cfg.image_dir, img_id, cfg.image_ext)
        ok = os.path.exists(p)
        keep_mask.append(ok)
        if not ok:
            missing.append(int(img_id))
    out = df.loc[keep_mask].copy()

    # Save a short missing report for debugging.
    _ensure_dir(cfg.output_dir)
    miss_path = os.path.join(cfg.output_dir, "missing_images_report.csv")
    pd.DataFrame({"missing_image_id": missing}).to_csv(miss_path, index=False)

    return out


def make_price_classes(cfg: PreprocessConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates `cfg.label_col` using quantile bins (tertiles) by default.
    """
    if cfg.num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    prices = df[cfg.price_col].values.astype(np.float64)

    if cfg.bin_method == "quantile":
        # Quantile bin edges: e.g., for 3 classes -> [0%, 33.33%, 66.67%, 100%]
        edges = np.quantile(prices, q=np.linspace(0.0, 1.0, cfg.num_classes + 1))
    elif cfg.bin_method == "uniform":
        edges = np.linspace(np.min(prices), np.max(prices), cfg.num_classes + 1)
    else:
        raise ValueError("bin_method must be 'quantile' or 'uniform'")

    # If many repeated price values lead to non-unique edges, pandas cut can fail.
    edges = np.unique(edges)
    if len(edges) - 1 < 2:
        raise ValueError("Could not create multiple bins; price values may be degenerate.")

    num_bins = len(edges) - 1
    labels = list(range(num_bins))

    df = df.copy()
    df[cfg.label_col] = pd.cut(
        df[cfg.price_col],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(int)

    # If the dataset collapses bins due to repeated values, warn in output.
    if num_bins != cfg.num_classes:
        stats = {
            "requested_num_classes": cfg.num_classes,
            "effective_num_classes": num_bins,
            "bin_method": cfg.bin_method,
            "bin_edges": edges.tolist(),
        }
        with open(os.path.join(cfg.output_dir, "label_binning_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        with open(os.path.join(cfg.output_dir, "label_binning_stats.json"), "w") as f:
            json.dump({"requested_num_classes": cfg.num_classes, "effective_num_classes": cfg.num_classes}, f, indent=2)

    return df


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/val/test with label proportions preserved.
    Uses sklearn if available; otherwise falls back to a simple (non-stratified) split.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(df) * test_frac)
        n_val = int(len(df) * val_frac)
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

    train_val_frac = 1.0 - (val_frac + test_frac)
    if train_val_frac <= 0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    # First split off test.
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        stratify=df[label_col],
    )

    # Split train/val from remaining.
    # Adjust val fraction relative to trainval.
    val_rel = val_frac / (val_frac + train_val_frac)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_rel,
        random_state=seed,
        stratify=df_trainval[label_col],
    )
    return df_train, df_val, df_test


def compute_tabular_normalization_stats(cfg: PreprocessConfig, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in cfg.tabular_cols:
        if col not in df.columns:
            continue
        col_vals = df[col].astype(float).values
        stats[col] = {"mean": float(np.mean(col_vals)), "std": float(np.std(col_vals) + 1e-12)}
    return stats


def save_split_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def build_transforms(
    image_size: int = 224,
    mean: List[float] = DEFAULT_IMAGE_MEAN,
    std: List[float] = DEFAULT_IMAGE_STD,
) -> Tuple[object, object]:
    """
    Returns (train_transform, val_transform).
    These are standard ImageNet normalization transforms so torchvision pretrained CNN/ViT works.
    """
    if T is None:
        raise ModuleNotFoundError("torch/torchvision not installed. Install PyTorch + torchvision to use transforms.")

    train_tfm = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    val_tfm = T.Compose(
        [
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfm, val_tfm


class HouseImagePriceDataset(Dataset):
    """
    Minimal image+label dataset.
    For later steps, you can extend __getitem__ to also return normalized tabular features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        image_ext: str,
        transform=None,
        image_id_col: str = "image_id",
        label_col: str = "price_class",
        tabular_cols: Tuple[str, ...] = ("bed", "bath", "sqft", "n_citi"),
    ):
        if Image is None:
            raise ModuleNotFoundError("PIL not available. Install pillow (usually comes with torchvision).")

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.transform = transform
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.tabular_cols = tabular_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = int(row[self.image_id_col])
        img_path = _image_path(self.image_dir, image_id, self.image_ext)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = int(row[self.label_col])
        return img, label


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PreprocessConfig,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    mean: List[float] = DEFAULT_IMAGE_MEAN,
    std: List[float] = DEFAULT_IMAGE_STD,
) -> Tuple[object, object, object]:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    if torch is None:
        raise ModuleNotFoundError("torch not installed. Install PyTorch to build DataLoaders.")

    train_tfm, val_tfm = build_transforms(image_size=image_size, mean=mean, std=std)

    train_ds = HouseImagePriceDataset(
        train_df,
        image_dir=cfg.image_dir,
        image_ext=cfg.image_ext,
        transform=train_tfm,
        image_id_col=cfg.image_id_col,
        label_col=cfg.label_col,
        tabular_cols=cfg.tabular_cols,
    )
    val_ds = HouseImagePriceDataset(
        val_df,
        image_dir=cfg.image_dir,
        image_ext=cfg.image_ext,
        transform=val_tfm,
        image_id_col=cfg.image_id_col,
        label_col=cfg.label_col,
        tabular_cols=cfg.tabular_cols,
    )
    test_ds = HouseImagePriceDataset(
        test_df,
        image_dir=cfg.image_dir,
        image_ext=cfg.image_ext,
        transform=val_tfm,
        image_id_col=cfg.image_id_col,
        label_col=cfg.label_col,
        tabular_cols=cfg.tabular_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def run_preprocessing_only(cfg: PreprocessConfig) -> Dict[str, object]:
    _ensure_dir(cfg.output_dir)

    df = load_and_clean_metadata(cfg)
    df = filter_rows_with_existing_images(cfg, df)
    df = make_price_classes(cfg, df)

    train_df, val_df, test_df = stratified_split(
        df,
        label_col=cfg.label_col,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )

    save_split_csv(train_df, os.path.join(cfg.output_dir, "train_split.csv"))
    save_split_csv(val_df, os.path.join(cfg.output_dir, "val_split.csv"))
    save_split_csv(test_df, os.path.join(cfg.output_dir, "test_split.csv"))

    tab_stats = compute_tabular_normalization_stats(cfg, train_df)
    with open(os.path.join(cfg.output_dir, "tabular_normalization.json"), "w") as f:
        json.dump(tab_stats, f, indent=2)

    # Helpful summary for debugging.
    summary = {
        "total_rows_after_image_filter": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_col": cfg.label_col,
        "label_counts_total": df[cfg.label_col].value_counts(sort=False).to_dict(),
        "label_counts_train": train_df[cfg.label_col].value_counts(sort=False).to_dict(),
        "label_counts_val": val_df[cfg.label_col].value_counts(sort=False).to_dict(),
        "label_counts_test": test_df[cfg.label_col].value_counts(sort=False).to_dict(),
        "bin_method": cfg.bin_method,
        "seed": cfg.seed,
    }
    with open(os.path.join(cfg.output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="socal2.csv")
    p.add_argument("--image_dir", type=str, default=os.path.join("socal2", "socal_pics"))
    p.add_argument("--output_dir", type=str, default=os.path.join("asgn2_artifacts", "step3"))
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--bin_method", type=str, default="quantile", choices=["quantile", "uniform"])
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        bin_method=args.bin_method,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    print("Running preprocessing + augmentation pipeline setup (split creation + artifacts)...")
    summary = run_preprocessing_only(cfg)
    print("Split summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

