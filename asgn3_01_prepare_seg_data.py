"""
Assignment 3 - Step 1
Prepare semantic-segmentation data and run sanity checks.

Expected raw data layout (after you download a labeled segmentation dataset):
  <dataset_root>/
    images/
      xxx.jpg (or .png/.jpeg/.bmp/.webp)
    masks/
      xxx.png  (single-channel class-id mask)

This script:
1) Matches images and masks by filename stem.
2) Validates dimensions and mask format.
3) Audits class IDs and class-pixel frequencies.
4) Creates train/val/test split CSVs.
5) Saves sanity plots: image | mask | overlay.
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("matplotlib is required. Install with `pip install matplotlib`.") from e


ALLOWED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_IGNORE_INDEX = 255


@dataclass(frozen=True)
class SegPrepConfig:
    dataset_root: str
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    output_dir: str = os.path.join("asgn3_artifacts", "step1")

    # Splits
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # If your masks use 255 for unlabeled/void, keep this as 255.
    ignore_index: int = DEFAULT_IGNORE_INDEX

    # Visualization
    sanity_samples: int = 12


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(path: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(path)):
        if os.path.splitext(name)[1].lower() in ALLOWED_IMAGE_EXTS:
            out.append(name)
    return out


def file_stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def build_pairs(cfg: SegPrepConfig) -> pd.DataFrame:
    images_dir = os.path.join(cfg.dataset_root, cfg.images_subdir)
    masks_dir = os.path.join(cfg.dataset_root, cfg.masks_subdir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    image_files = list_images(images_dir)
    mask_files = list_images(masks_dir)

    image_map = {file_stem(x): x for x in image_files}
    mask_map = {file_stem(x): x for x in mask_files}

    common = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))

    rows = []
    for stem in common:
        rows.append(
            {
                "id": stem,
                "image_path": os.path.join(images_dir, image_map[stem]),
                "mask_path": os.path.join(masks_dir, mask_map[stem]),
            }
        )

    df = pd.DataFrame(rows)
    ensure_dir(cfg.output_dir)

    pd.DataFrame({"missing_mask_for_image_id": missing_masks}).to_csv(
        os.path.join(cfg.output_dir, "missing_masks.csv"),
        index=False,
    )
    pd.DataFrame({"missing_image_for_mask_id": missing_images}).to_csv(
        os.path.join(cfg.output_dir, "missing_images.csv"),
        index=False,
    )
    return df


def validate_and_profile_masks(cfg: SegPrepConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Keeps only rows with valid image/mask pairs:
    - same HxW
    - mask is single-channel after conversion (P/L/I modes)
    """
    valid_rows = []
    invalid_rows = []
    class_pixel_counts: Dict[int, int] = {}

    for _, row in df.iterrows():
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path)

            # Segmentation masks should be single-channel class IDs.
            # Mode "P", "L", "I" are typical; "RGB" masks are ambiguous here.
            if mask_pil.mode == "RGB":
                raise ValueError(
                    "Mask is RGB. Expected class-id masks (single channel). "
                    "Convert RGB mask colors to class IDs first."
                )

            mask = np.array(mask_pil)
            if mask.ndim != 2:
                raise ValueError(f"Mask ndim={mask.ndim}; expected 2D class-id map.")

            if img.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(
                    f"Image/mask size mismatch: image={img.size}, mask={(mask.shape[1], mask.shape[0])}"
                )

            # Count class IDs (excluding ignore index).
            unique, counts = np.unique(mask, return_counts=True)
            for cls_id, c in zip(unique.tolist(), counts.tolist()):
                if int(cls_id) == cfg.ignore_index:
                    continue
                class_pixel_counts[int(cls_id)] = class_pixel_counts.get(int(cls_id), 0) + int(c)

            valid_rows.append(row.to_dict())
        except Exception as e:  # broad by design: captures file/shape/mode issues.
            invalid_rows.append(
                {
                    "id": row["id"],
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "error": str(e),
                }
            )

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    invalid_df.to_csv(os.path.join(cfg.output_dir, "invalid_pairs.csv"), index=False)

    # Dominant class for rough split stratification.
    dominant_class = []
    for _, row in valid_df.iterrows():
        mask = np.array(Image.open(row["mask_path"]))
        unique, counts = np.unique(mask, return_counts=True)
        pairs = [(int(u), int(c)) for u, c in zip(unique.tolist(), counts.tolist()) if int(u) != cfg.ignore_index]
        if len(pairs) == 0:
            dominant_class.append(-1)
        else:
            pairs.sort(key=lambda x: x[1], reverse=True)
            dominant_class.append(pairs[0][0])
    valid_df["dominant_class"] = dominant_class

    profile = {
        "valid_pairs": int(len(valid_df)),
        "invalid_pairs": int(len(invalid_df)),
        "ignore_index": int(cfg.ignore_index),
        "class_pixel_counts": {str(k): int(v) for k, v in sorted(class_pixel_counts.items())},
        "observed_class_ids": sorted([int(x) for x in class_pixel_counts.keys()]),
    }
    return valid_df, profile


def split_df(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
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
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()

    # Try stratified split by dominant_class when possible.
    # CMP can have very rare dominant classes (count=1), which breaks sklearn stratify.
    def _can_stratify(series: pd.Series) -> bool:
        counts = series.value_counts(dropna=False)
        if len(counts) < 2:
            return False
        return int(counts.min()) >= 2

    stratify_full = df["dominant_class"] if _can_stratify(df["dominant_class"]) else None

    # First split off test.
    train_val, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        stratify=stratify_full,
    )

    rel_val = val_frac / (1.0 - test_frac)
    stratify_trainval = (
        train_val["dominant_class"] if _can_stratify(train_val["dominant_class"]) else None
    )
    train_df, val_df = train_test_split(
        train_val,
        test_size=rel_val,
        random_state=seed,
        stratify=stratify_trainval,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Deterministic pseudo-colormap for class IDs.
    """
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_ids = np.unique(mask)
    for cls_id in unique_ids.tolist():
        if cls_id == DEFAULT_IGNORE_INDEX:
            color = (0, 0, 0)
        else:
            # Simple deterministic palette based on class id.
            color = ((37 * int(cls_id)) % 255, (67 * int(cls_id)) % 255, (97 * int(cls_id)) % 255)
        colored[mask == cls_id] = color
    return colored


def save_sanity_plots(cfg: SegPrepConfig, df: pd.DataFrame, title: str) -> None:
    if len(df) == 0:
        return
    out_dir = os.path.join(cfg.output_dir, "sanity_plots")
    ensure_dir(out_dir)

    n = min(cfg.sanity_samples, len(df))
    sample_df = df.sample(n=n, random_state=cfg.seed).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]))
        c_mask = colorize_mask(mask)
        overlay = (0.6 * img + 0.4 * c_mask).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[1].imshow(c_mask)
        axes[1].set_title("Mask (colorized)")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{title} | id={row['id']}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:03d}_{row['id']}.png"), dpi=140, bbox_inches="tight")
        plt.close()


def write_summary(cfg: SegPrepConfig, profile: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    summary = {
        "dataset_root": cfg.dataset_root,
        "images_subdir": cfg.images_subdir,
        "masks_subdir": cfg.masks_subdir,
        "valid_pairs": profile["valid_pairs"],
        "invalid_pairs": profile["invalid_pairs"],
        "observed_class_ids": profile["observed_class_ids"],
        "class_pixel_counts": profile["class_pixel_counts"],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_frac": cfg.val_frac,
        "test_frac": cfg.test_frac,
        "seed": cfg.seed,
    }
    with open(os.path.join(cfg.output_dir, "step1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, required=True, help="Root containing images/ and masks/")
    p.add_argument("--images_subdir", type=str, default="images")
    p.add_argument("--masks_subdir", type=str, default="masks")
    p.add_argument("--output_dir", type=str, default=os.path.join("asgn3_artifacts", "step1"))
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore_index", type=int, default=DEFAULT_IGNORE_INDEX)
    p.add_argument("--sanity_samples", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegPrepConfig(
        dataset_root=args.dataset_root,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        ignore_index=args.ignore_index,
        sanity_samples=args.sanity_samples,
    )
    ensure_dir(cfg.output_dir)

    print("[Step 1] Matching images and masks...")
    pairs_df = build_pairs(cfg)
    pairs_df.to_csv(os.path.join(cfg.output_dir, "all_pairs.csv"), index=False)

    print("[Step 1] Validating mask format + dimensions + class IDs...")
    valid_df, profile = validate_and_profile_masks(cfg, pairs_df)
    valid_df.to_csv(os.path.join(cfg.output_dir, "valid_pairs.csv"), index=False)

    print("[Step 1] Creating train/val/test split...")
    train_df, val_df, test_df = split_df(valid_df, cfg.val_frac, cfg.test_frac, cfg.seed)
    train_df.to_csv(os.path.join(cfg.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.output_dir, "test_split.csv"), index=False)

    print("[Step 1] Saving sanity plots...")
    save_sanity_plots(cfg, train_df, title="train")

    write_summary(cfg, profile, train_df, val_df, test_df)
    print(f"Done. Outputs written to: {cfg.output_dir}")
    print("Read step1_summary.json and inspect sanity_plots/ before training.")


if __name__ == "__main__":
    main()

