"""
Assignment 3 — per-run segmentation evaluation + qualitative panels.

For each `asgn3_runs/<name>/` with `train_config.json`, `class_mapping.json`, `best_model.pt`:

- Pixel confusion matrix (true vs predicted class, ignoring void if any)
- Per-class IoU bar chart + global mIoU
- `best_k` / `worst_k` image panels (by per-image mean IoU over present classes)

Usage:
  python asgn3_03_visualize_seg.py --run_dir asgn3_runs/d_deeplab_pretrained_finetuned
  python asgn3_03_visualize_seg.py --runs_root asgn3_runs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("matplotlib required: pip install matplotlib") from e

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("torch required") from e

from asgn3_02_train_seg import (
    DEFAULT_IGNORE_INDEX,
    SegmentationDataset,
    build_lookup_table,
    build_seg_model,
    forward_logits,
)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def colorize_cls_map(arr: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        out[arr == c] = ((37 * c) % 255, (67 * c) % 255, (97 * c) % 255)
    return out


def mean_iou_image(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int) -> float:
    valid = gt != ignore_index
    if not np.any(valid):
        return 0.0
    gt_v = gt[valid]
    pr_v = pred[valid]

    present = []
    for c in range(num_classes):
        inter = np.sum((gt_v == c) & (pr_v == c))
        denom = np.sum((gt_v == c) | (pr_v == c))
        if denom > 0:
            present.append(inter / denom)
    if len(present) == 0:
        return 0.0
    return float(np.mean(present))


def confusion_from_pair(pred: np.ndarray, gt: np.ndarray, k: int, ignore_index: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    valid = gt != ignore_index
    if not np.any(valid):
        return cm
    gt_v = gt[valid].astype(np.int64).ravel()
    pr_v = pred[valid].astype(np.int64).ravel()
    idx = gt_v * k + pr_v
    cm += np.bincount(idx, minlength=k * k).reshape(k, k)
    return cm


def per_class_iou_from_cm(cm: np.ndarray) -> np.ndarray:
    k = cm.shape[0]
    ious = np.zeros(k, dtype=np.float64)
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious[c] = tp / denom if denom > 0 else 0.0
    return ious


def plot_confusion(cm: np.ndarray, title: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(title + " (counts)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_xlabel("Pred")
    axes[0].set_ylabel("True")

    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cn = cm.astype(np.float64) / row_sum
    im1 = axes[1].imshow(cn, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(title + " (row-normalized)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_xlabel("Pred")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_panels(rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]], out_path: str, suptitle: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (img_rgb, gt_col, pr_col, title) in enumerate(rows):
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(title, fontsize=8)
        axes[i, 1].imshow(gt_col)
        axes[i, 1].set_title("GT colorized")
        axes[i, 2].imshow(pr_col)
        axes[i, 2].set_title("Pred colorized")
        for j in range(3):
            axes[i, j].axis("off")
    plt.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_one_run(run_dir: str, top_k: int = 3, artifacts_override: Optional[str] = None) -> str:
    run_dir = os.path.abspath(run_dir)
    viz_dir = os.path.join(run_dir, "viz_eval")
    os.makedirs(viz_dir, exist_ok=True)

    tc = load_json(os.path.join(run_dir, "train_config.json"))
    mapping = load_json(os.path.join(run_dir, "class_mapping.json"))

    artifacts_dir = artifacts_override or tc["artifacts_dir"]
    test_df = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv"))

    raw_ids = mapping["raw_class_ids"]
    num_classes = int(mapping["num_classes"])
    ignore_index = int(mapping.get("ignore_index", DEFAULT_IGNORE_INDEX))
    lut = build_lookup_table(raw_ids, ignore_index=ignore_index)

    ds = SegmentationDataset(
        test_df,
        image_size=int(tc["image_size"]),
        lut=lut,
        ignore_index=ignore_index,
        train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    model = build_seg_model(
        tc["model"],
        num_classes=num_classes,
        pretrained=bool(tc.get("pretrained", False)),
        freeze_backbone=bool(tc.get("freeze_backbone", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model_name = str(tc["model"])

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_ious: List[float] = []

    with torch.no_grad():
        for i in range(len(ds)):
            x, y_tensor = ds[i]
            xb = x.unsqueeze(0).to(device)
            logits = forward_logits(model, xb, model_name=model_name)
            if logits.shape[-2:] != y_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=y_tensor.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            gt = y_tensor.numpy().astype(np.int64)

            cm += confusion_from_pair(pred, gt, num_classes, ignore_index)
            per_ious.append(mean_iou_image(pred, gt, num_classes, ignore_index))

    per_ious_np = np.array(per_ious, dtype=np.float64)
    ious = per_class_iou_from_cm(cm)
    miou = float(np.mean(ious))

    run_title = os.path.basename(run_dir.rstrip("/"))
    plot_confusion(cm, title=run_title, out_path=os.path.join(viz_dir, "confusion_pixels.png"))

    plt.figure(figsize=(10, 3))
    plt.bar(range(num_classes), ious)
    plt.ylim(0, 1.05)
    plt.xlabel("Class (remapped 0..K-1)")
    plt.ylabel("IoU")
    plt.title(f"{run_title}: per-class IoU (global mIoU={miou:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "per_class_iou.png"), dpi=150)
    plt.close()

    idx_best = np.argsort(-per_ious_np)
    idx_worst = np.argsort(per_ious_np)

    def build_row(sample_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        row = test_df.iloc[sample_idx]
        stem = row.get("id", sample_idx)

        pil_img = Image.open(row["image_path"]).convert("RGB")
        img_np = np.array(pil_img)
        gt_raw = np.array(Image.open(row["mask_path"]))
        gt = lut[gt_raw]
        gt_col = colorize_cls_map(gt, num_classes)

        xb, yt = ds[sample_idx]
        with torch.no_grad():
            xb = xb.unsqueeze(0).to(device)
            logits = forward_logits(model, xb, model_name=model_name)
            logits = F.interpolate(logits, size=yt.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        pred_col = colorize_cls_map(pred, num_classes)

        title = f"id={stem}  meanIoU={per_ious_np[sample_idx]:.3f}"
        return img_np, gt_col, pred_col, title

    best_rows = [build_row(int(i)) for i in idx_best[:top_k]]
    worst_rows = [build_row(int(i)) for i in idx_worst[:top_k]]

    plot_panels(
        best_rows,
        os.path.join(viz_dir, f"best_{top_k}_segmentations.png"),
        suptitle=f"{run_title}: highest mean IoU (test)",
    )
    plot_panels(
        worst_rows,
        os.path.join(viz_dir, f"worst_{top_k}_segmentations.png"),
        suptitle=f"{run_title}: lowest mean IoU (test)",
    )

    summary = {
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "num_test": int(len(test_df)),
        "miou_pixels": miou,
        "per_class_iou": ious.tolist(),
        "mean_mean_iou_per_image": float(per_ious_np.mean()),
        "best_indices": idx_best[:top_k].tolist(),
        "worst_indices": idx_worst[:top_k].tolist(),
    }
    with open(os.path.join(viz_dir, "viz_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return viz_dir


def find_runs(root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if all(
            os.path.isfile(os.path.join(d, fn))
            for fn in ("train_config.json", "class_mapping.json", "best_model.pt")
        ):
            out.append(d)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--runs_root", type=str, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override test_split location")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        out = visualize_one_run(args.run_dir, top_k=args.top_k, artifacts_override=args.artifacts_dir)
        print(f"Saved: {out}")
        return
    if args.runs_root:
        for d in find_runs(args.runs_root):
            print(f"Processing {d} ...")
            out = visualize_one_run(d, top_k=args.top_k, artifacts_override=args.artifacts_dir)
            print(f"  -> {out}")
        return
    raise SystemExit("Provide --run_dir or --runs_root")


if __name__ == "__main__":
    main()
