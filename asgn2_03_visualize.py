"""
Assignment 2 - Visualizations from trained `asgn2_runs/*` checkpoints.

For each run directory (with `train_config.json` + `best_model.pt`):
- Confusion matrix (counts + normalized)
- 3 "best" predictions: correct with highest confidence on the true class
- 3 "worst" predictions: incorrect with highest confidence on the wrong predicted class
- Per-class precision / recall / F1 bar chart
- Histogram of max softmax probability for correct vs incorrect predictions

Usage:
  python3 asgn2_03_visualize.py --run_dir asgn2_runs/a_custom_cnn
  python3 asgn2_03_visualize.py --runs_root asgn2_runs   # all subfolders with best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from asgn2_01_preprocess_augment import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    PreprocessConfig,
    build_transforms,
)
from asgn2_models import build_model

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch is required. pip install torch torchvision pillow matplotlib"
    ) from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("matplotlib is required. pip install matplotlib") from e

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ModuleNotFoundError:
    confusion_matrix = None
    classification_report = None
    precision_recall_fscore_support = None


def _image_path(image_dir: str, image_id: int, image_ext: str) -> str:
    return os.path.join(image_dir, f"{int(image_id)}{image_ext}")


class TestDatasetWithId(Dataset):
    """Same as training test set, but returns (tensor, label, image_id) for plotting."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        image_ext: str,
        transform,
        image_id_col: str = "image_id",
        label_col: str = "price_class",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.transform = transform
        self.image_id_col = image_id_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = int(row[self.image_id_col])
        path = _image_path(self.image_dir, image_id, self.image_ext)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label, image_id


def load_run_config(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "train_config.json")
    with open(path) as f:
        return json.load(f)


def collect_test_predictions(
    run_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      y_true, y_pred, prob_true, prob_pred, max_prob, meta (dict with class names etc.)
    """
    cfg = load_run_config(run_dir)
    artifacts_dir = cfg["artifacts_dir"]
    image_dir = cfg["image_dir"]
    image_size = int(cfg.get("image_size", 224))

    test_df = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv"))
    _, val_tfm = build_transforms(
        image_size=image_size,
        mean=DEFAULT_IMAGE_MEAN,
        std=DEFAULT_IMAGE_STD,
    )

    preprocess_cfg = PreprocessConfig(
        csv_path=cfg["csv_path"],
        image_dir=image_dir,
        output_dir=artifacts_dir,
        num_classes=int(cfg.get("num_classes", 3)),
    )

    ds = TestDatasetWithId(
        test_df,
        image_dir=image_dir,
        image_ext=preprocess_cfg.image_ext,
        transform=val_tfm,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=cfg["model"],
        num_classes=int(cfg["num_classes"]),
        pretrained=bool(cfg.get("pretrained", False)),
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
    ).to(device)

    ckpt_path = os.path.join(run_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob_true: List[float] = []
    all_prob_pred: List[float] = []
    all_max_prob: List[float] = []
    all_image_ids: List[int] = []

    with torch.no_grad():
        for x, y, image_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            for i in range(x.size(0)):
                t = int(y[i].item())
                p = int(pred[i].item())
                pr = probs[i]
                all_true.append(t)
                all_pred.append(p)
                all_prob_true.append(float(pr[t].item()))
                all_prob_pred.append(float(pr[p].item()))
                all_max_prob.append(float(pr.max().item()))
                all_image_ids.append(int(image_ids[i].item()))

    num_classes = int(cfg["num_classes"])
    meta = {
        "num_classes": num_classes,
        "model": cfg["model"],
        "class_names": [f"Class {k}" for k in range(num_classes)],
    }
    # Optional human-readable names for 3-class price bins
    if num_classes == 3:
        meta["class_names"] = ["Low price", "Mid price", "High price"]

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob_true),
        np.array(all_prob_pred),
        np.array(all_max_prob),
        meta,
    )


def _sk_cm(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"{title} (counts)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title(f"{title} (row-normalized)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
) -> None:
    if precision_recall_fscore_support is None:
        return
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, p, width=w, label="Precision")
    ax.bar(x, r, width=w, label="Recall")
    ax.bar(x + w, f1, width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_prob: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        max_prob[correct],
        bins=30,
        alpha=0.6,
        label=f"Correct (n={correct.sum()})",
        color="green",
        density=True,
    )
    ax.hist(
        max_prob[~correct],
        bins=30,
        alpha=0.6,
        label=f"Incorrect (n={(~correct).sum()})",
        color="red",
        density=True,
    )
    ax.set_xlabel("Max softmax probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_example_grid(
    run_dir: str,
    indices: List[int],
    test_df: pd.DataFrame,
    image_dir: str,
    image_ext: str,
    titles: List[str],
    out_path: str,
    suptitle: str,
    image_size: int = 224,
) -> None:
    """Load raw images (no augment) for display."""
    n = len(indices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, idx, title in zip(axes[:n], indices, titles):
        row = test_df.iloc[idx]
        image_id = int(row["image_id"])
        path = _image_path(image_dir, image_id, image_ext)
        img = Image.open(path).convert("RGB")
        img.thumbnail((image_size * 2, image_size * 2))
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_run(run_dir: str, batch_size: int = 64, num_workers: int = 2, top_k: int = 3) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.join(run_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_run_config(run_dir)
    artifacts_dir = cfg["artifacts_dir"]
    test_df = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv")).reset_index(drop=True)

    y_true, y_pred, prob_true, prob_pred, max_prob, meta = collect_test_predictions(
        run_dir, batch_size=batch_size, num_workers=num_workers
    )
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]

    cm = _sk_cm(y_true, y_pred, num_classes)
    run_name = os.path.basename(run_dir.rstrip("/"))
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(out_dir, "confusion_matrix.png"),
        title=run_name,
    )

    plot_per_class_metrics(
        y_true,
        y_pred,
        class_names,
        os.path.join(out_dir, "per_class_precision_recall_f1.png"),
        title=f"{run_name}: per-class metrics (test)",
    )

    plot_confidence_histogram(
        y_true,
        y_pred,
        max_prob,
        os.path.join(out_dir, "confidence_histogram_correct_vs_wrong.png"),
        title=f"{run_name}: confidence on test set",
    )

    # Best: correct, highest P(true class)
    correct_mask = y_true == y_pred
    if correct_mask.any():
        n_best = min(top_k, int(correct_mask.sum()))
        best_idx = np.argsort(-prob_true[correct_mask])[:n_best]
        global_idx = np.where(correct_mask)[0][best_idx]
    else:
        global_idx = np.array([], dtype=np.int64)

    # Worst: incorrect, highest P(predicted) = confident mistakes
    wrong_mask = ~correct_mask
    if wrong_mask.any():
        n_worst = min(top_k, int(wrong_mask.sum()))
        worst_idx = np.argsort(-prob_pred[wrong_mask])[:n_worst]
        global_idx_w = np.where(wrong_mask)[0][worst_idx]
    else:
        global_idx_w = np.array([], dtype=np.int64)

    image_dir = cfg["image_dir"]
    preprocess_cfg = PreprocessConfig(
        csv_path=cfg["csv_path"],
        image_dir=image_dir,
        output_dir=artifacts_dir,
        num_classes=num_classes,
    )

    titles_best = []
    for gi in global_idx:
        t, p = int(y_true[gi]), int(y_pred[gi])
        titles_best.append(
            f"id={test_df.iloc[gi]['image_id']}\nTrue={class_names[t]}  Pred={class_names[p]}\nP(true)={prob_true[gi]:.3f}"
        )

    titles_worst = []
    for gi in global_idx_w:
        t, p = int(y_true[gi]), int(y_pred[gi])
        titles_worst.append(
            f"id={test_df.iloc[gi]['image_id']}\nTrue={class_names[t]}  Pred={class_names[p]}\nP(pred)={prob_pred[gi]:.3f}"
        )

    if len(global_idx) > 0:
        plot_example_grid(
            run_dir,
            global_idx.tolist(),
            test_df,
            image_dir,
            preprocess_cfg.image_ext,
            titles_best,
            os.path.join(out_dir, f"best_{top_k}_predictions.png"),
            suptitle=f"{run_name}: {top_k} most confident *correct* predictions (test)",
            image_size=int(cfg.get("image_size", 224)),
        )

    if len(global_idx_w) > 0:
        plot_example_grid(
            run_dir,
            global_idx_w.tolist(),
            test_df,
            image_dir,
            preprocess_cfg.image_ext,
            titles_worst,
            os.path.join(out_dir, f"worst_{top_k}_predictions.png"),
            suptitle=f"{run_name}: {top_k} most confident *wrong* predictions (test)",
            image_size=int(cfg.get("image_size", 224)),
        )

    report = None
    if classification_report is not None:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )

    summary = {
        "run_dir": run_dir,
        "test_accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_prediction_indices": global_idx.tolist(),
        "worst_prediction_indices": global_idx_w.tolist(),
    }
    with open(os.path.join(out_dir, "viz_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if report:
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(report)

    return out_dir


def find_run_dirs(runs_root: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "best_model.pt")) and os.path.isfile(
            os.path.join(d, "train_config.json")
        ):
            out.append(d)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None, help="Single run folder under asgn2_runs/...")
    p.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Process every subfolder that contains best_model.pt + train_config.json",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        out = visualize_run(
            args.run_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
        )
        print(f"Saved visualizations to: {out}")
        return
    if args.runs_root:
        for d in find_run_dirs(args.runs_root):
            print(f"Processing {d} ...")
            out = visualize_run(
                d,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                top_k=args.top_k,
            )
            print(f"  -> {out}")
        return
    raise SystemExit("Provide --run_dir or --runs_root")


if __name__ == "__main__":
    main()
