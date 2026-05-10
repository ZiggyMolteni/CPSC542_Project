"""
Assignment 3 - Step 2  (robust v2)
Train semantic segmentation models with reusable, modular pipeline.

Improvements over v1:
  - Class-weighted cross-entropy loss (--use_class_weights) to address label imbalance
  - Mixed-precision training (--use_amp) via torch.cuda.amp for faster GPU training
  - Extra augmentations: random rotation + random scale crop (image+mask aligned)
  - Per-class IoU logged every epoch in history.json
  - DeepLabV3+ ResNet101 added as model option
  - torch.load weights_only=True to silence FutureWarning

Consumes artifacts from Step 1:
  asgn3_artifacts/step1/{train,val,test}_split.csv
  asgn3_artifacts/step1/step1_summary.json  (for class pixel counts)

Models:
  - custom_unet               (your own small U-Net from scratch; regime (a))
  - deeplabv3_resnet50        (CNN-based)
  - deeplabv3_resnet101       (stronger CNN backbone)
  - fcn_resnet50              (CNN-based)
  - segformer_b0              (Transformer-based, requires `transformers`)

Outputs:
  - run_dir/train_config.json
  - run_dir/class_mapping.json
  - run_dir/history.json
  - run_dir/best_model.pt
  - run_dir/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    import torchvision.transforms.functional as TF
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch/torchvision is required. Install with `pip install -r requirements.txt`."
    ) from e


DEFAULT_IGNORE_INDEX = 255


@dataclass(frozen=True)
class SegTrainConfig:
    artifacts_dir: str = os.path.join("asgn3_artifacts", "step1")
    run_dir: str = os.path.join("asgn3_runs", "debug_seg")

    model: str = "deeplabv3_resnet50"
    pretrained: bool = False
    freeze_backbone: bool = False

    image_size: int = 384
    batch_size: int = 4
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    ignore_index: int = DEFAULT_IGNORE_INDEX

    # Robustness flags
    use_class_weights: bool = False   # inverse-frequency weighting in loss
    use_amp: bool = False             # mixed-precision training (torch.cuda.amp)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(artifacts_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(artifacts_dir, "train_split.csv"))
    val_df   = pd.read_csv(os.path.join(artifacts_dir, "val_split.csv"))
    test_df  = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv"))
    return train_df, val_df, test_df


def infer_class_ids_from_masks(
    dfs: List[pd.DataFrame],
    ignore_index: int,
) -> List[int]:
    class_ids: set = set()
    for df in dfs:
        for p in df["mask_path"].tolist():
            m = np.array(Image.open(p))
            for i in np.unique(m).tolist():
                i = int(i)
                if i != ignore_index:
                    class_ids.add(i)
    return sorted(class_ids)


def build_lookup_table(class_ids: List[int], ignore_index: int) -> np.ndarray:
    """Maps raw mask IDs -> contiguous [0..K-1]; unknown -> ignore_index."""
    lut = np.full(256, ignore_index, dtype=np.uint8)
    for new_id, raw_id in enumerate(class_ids):
        if raw_id < 0 or raw_id > 255:
            raise ValueError(f"Raw class id out of [0,255] range: {raw_id}")
        lut[raw_id] = new_id
    return lut


def compute_class_weights(
    train_df: pd.DataFrame,
    class_ids: List[int],
    lut: np.ndarray,
    ignore_index: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency weights for each contiguous class ID.
    Reads pixel counts from step1_summary.json if available (fast path),
    otherwise scans all training masks directly.
    """
    num_classes = len(class_ids)
    counts = np.zeros(num_classes, dtype=np.float64)

    print("[Weights] Computing class pixel frequencies from training masks...")
    for p in train_df["mask_path"].tolist():
        m = np.array(Image.open(p), dtype=np.uint8)
        remapped = lut[m]
        for c in range(num_classes):
            counts[c] += int((remapped == c).sum())

    # Inverse frequency: w_c = total / (K * count_c), clipped to avoid extreme weights.
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.05, 20.0)
    print(f"[Weights] Per-class weights (remapped 0..{num_classes-1}): {np.round(weights, 3).tolist()}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset with extra augmentations
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        lut: np.ndarray,
        ignore_index: int,
        train: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.lut = lut
        self.ignore_index = ignore_index
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img  = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"])

        # --- Resize (nearest for mask to preserve class IDs) ---
        img  = TF.resize(img,  [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            # Horizontal flip
            if np.random.rand() < 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

            # Random rotation ±10°
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-10, 10))
                img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR,
                                 fill=0)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,
                                 fill=self.ignore_index)

            # Random scale crop: zoom into 80-100% of the image then resize back
            if np.random.rand() < 0.5:
                scale = float(np.random.uniform(0.80, 1.00))
                h = w = self.image_size
                new_h = int(h * scale)
                new_w = int(w * scale)
                top  = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                img  = TF.crop(img,  top, left, new_h, new_w)
                mask = TF.crop(mask, top, left, new_h, new_w)
                img  = TF.resize(img,  [h, w], interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, [h, w], interpolation=TF.InterpolationMode.NEAREST)

            # Color jitter (image only)
            if np.random.rand() < 0.5:
                img = TF.adjust_brightness(img, float(np.random.uniform(0.8, 1.2)))
                img = TF.adjust_contrast(img,   float(np.random.uniform(0.8, 1.2)))
                img = TF.adjust_saturation(img, float(np.random.uniform(0.8, 1.2)))

        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        m = np.array(mask, dtype=np.uint8)
        m = self.lut[m]
        y = torch.from_numpy(m.astype(np.int64))
        return x, y


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2: dh // 2 + x.shape[-2],
                              dw // 2: dw // 2 + x.shape[-1]]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CustomUNet(nn.Module):
    """Small U-Net for semantic segmentation (regime (a))."""

    def __init__(self, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.enc1      = DoubleConv(3, base_ch)
        self.enc2      = DoubleConv(base_ch,     base_ch * 2)
        self.enc3      = DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck= DoubleConv(base_ch * 4, base_ch * 8)
        self.pool      = nn.MaxPool2d(2)
        self.up1       = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2       = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3       = Up(base_ch * 2, base_ch,     base_ch)
        self.outc      = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x  = self.bottleneck(self.pool(s3))
        x  = self.up1(x, s3)
        x  = self.up2(x, s2)
        x  = self.up3(x, s1)
        return self.outc(x)


def _replace_deeplab_head(m: nn.Module, num_classes: int) -> None:
    in_ch = m.classifier[-1].in_channels
    m.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
        aux_in = m.aux_classifier[-1].in_channels
        m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)


def _replace_fcn_head(m: nn.Module, num_classes: int) -> None:
    in_ch = m.classifier[-1].in_channels
    m.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
        aux_in = m.aux_classifier[-1].in_channels
        m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)


def build_seg_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    name = model_name.lower().strip()

    if name == "custom_unet":
        if pretrained:
            raise ValueError("custom_unet does not support pretrained weights.")
        m = CustomUNet(num_classes=num_classes)
        if freeze_backbone:
            for p in list(m.enc1.parameters()) + list(m.enc2.parameters()) + \
                     list(m.enc3.parameters()) + list(m.bottleneck.parameters()):
                p.requires_grad = False
        return m

    if name == "deeplabv3_resnet50":
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        if pretrained:
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
            _replace_deeplab_head(m, num_classes)
        else:
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    if name == "deeplabv3_resnet101":
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        if pretrained:
            m = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
            _replace_deeplab_head(m, num_classes)
        else:
            m = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, num_classes=num_classes)
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    if name == "fcn_resnet50":
        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
        if pretrained:
            m = torchvision.models.segmentation.fcn_resnet50(weights=weights)
            _replace_fcn_head(m, num_classes)
        else:
            m = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    if name == "segformer_b0":
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "segformer_b0 requires `transformers`. pip install transformers"
            ) from e
        if pretrained:
            m = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(num_labels=num_classes)
            m = SegformerForSemanticSegmentation(config)
        if freeze_backbone:
            for p in m.segformer.parameters():
                p.requires_grad = False
        return m

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Use: custom_unet, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, segformer_b0"
    )


def forward_logits(model: nn.Module, x: torch.Tensor, model_name: str) -> torch.Tensor:
    name = model_name.lower().strip()
    out  = model(x)
    if name in {"deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50"}:
        return out["out"]
    if name == "custom_unet":
        return out
    if name == "segformer_b0":
        return out.logits
    raise ValueError(f"Unsupported model_name: {model_name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    valid = y != ignore_index
    if valid.sum().item() == 0:
        return {"pixel_acc": 0.0, "miou": 0.0, "per_class_iou": [0.0] * num_classes}

    pixel_acc = float((preds[valid] == y[valid]).float().mean().item())

    per_class_iou: List[Optional[float]] = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        y_c    = (y    == c) & valid
        inter  = (pred_c & y_c).sum().item()
        union  = (pred_c | y_c).sum().item()
        per_class_iou.append(inter / union if union > 0 else None)

    valid_ious = [v for v in per_class_iou if v is not None]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    # Replace None with 0.0 for JSON serialisation
    per_class_iou_out = [v if v is not None else 0.0 for v in per_class_iou]
    return {"pixel_acc": pixel_acc, "miou": miou, "per_class_iou": per_class_iou_out}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def sanity_check_batch(loader, num_classes: int, ignore_index: int) -> None:
    x, y = next(iter(loader))
    assert x.ndim == 4 and y.ndim == 3, f"Unexpected shapes: x={x.shape}, y={y.shape}"
    assert x.shape[0] == y.shape[0], "Batch size mismatch between image and mask"
    assert x.shape[2:] == y.shape[1:], "Spatial mismatch between image and mask"
    y_np = y.numpy()
    bad  = ((y_np < 0) | ((y_np >= num_classes) & (y_np != ignore_index))).any()
    if bad:
        u = np.unique(y_np)
        raise ValueError(
            f"Mask contains unexpected IDs after remap. Unique: {u.tolist()}, "
            f"expected [0..{num_classes-1}] + ignore_index={ignore_index}"
        )


def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    model_name: str,
    num_classes: int,
    ignore_index: int,
    criterion: nn.Module,
    train: bool,
    scaler=None,
) -> Dict[str, float]:
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc  = 0.0
    total_miou = 0.0
    per_class_iou_accum = np.zeros(num_classes, dtype=np.float64)
    n = 0

    use_amp = scaler is not None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = forward_logits(model, x, model_name=model_name)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:],
                                           mode="bilinear", align_corners=False)
                loss = criterion(logits, y)
        else:
            logits = forward_logits(model, x, model_name=model_name)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:],
                                       mode="bilinear", align_corners=False)
            loss = criterion(logits, y)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            metrics = compute_batch_metrics(
                logits.float(), y, num_classes=num_classes, ignore_index=ignore_index
            )
        total_loss += float(loss.item())
        total_acc  += metrics["pixel_acc"]
        total_miou += metrics["miou"]
        per_class_iou_accum += np.array(metrics["per_class_iou"])
        n += 1

    denom = max(n, 1)
    return {
        "loss":          total_loss / denom,
        "pixel_acc":     total_acc  / denom,
        "miou":          total_miou / denom,
        "per_class_iou": (per_class_iou_accum / denom).tolist(),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(cfg: SegTrainConfig) -> Dict[str, float]:
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "train_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  |  AMP: {cfg.use_amp}  |  class weights: {cfg.use_class_weights}")

    train_df, val_df, test_df = load_splits(cfg.artifacts_dir)
    class_ids   = infer_class_ids_from_masks([train_df, val_df, test_df], ignore_index=cfg.ignore_index)
    num_classes = len(class_ids)
    lut         = build_lookup_table(class_ids, ignore_index=cfg.ignore_index)

    with open(os.path.join(cfg.run_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {
                "raw_class_ids": class_ids,
                "remapped_to":   {str(raw): i for i, raw in enumerate(class_ids)},
                "num_classes":   num_classes,
                "ignore_index":  cfg.ignore_index,
            },
            f, indent=2,
        )

    # Build loss (optionally class-weighted)
    if cfg.use_class_weights:
        w = compute_class_weights(train_df, class_ids, lut, cfg.ignore_index).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, ignore_index=cfg.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)

    # Datasets & loaders
    train_ds = SegmentationDataset(train_df, cfg.image_size, lut, cfg.ignore_index, train=True)
    val_ds   = SegmentationDataset(val_df,   cfg.image_size, lut, cfg.ignore_index, train=False)
    test_ds  = SegmentationDataset(test_df,  cfg.image_size, lut, cfg.ignore_index, train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)

    sanity_check_batch(train_loader, num_classes=num_classes, ignore_index=cfg.ignore_index)

    model = build_seg_model(
        cfg.model, num_classes=num_classes,
        pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler    = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    best_val_miou = -1.0
    best_path     = os.path.join(cfg.run_dir, "best_model.pt")
    history: List[Dict] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg.model,
                       num_classes, cfg.ignore_index, criterion, train=True, scaler=scaler)
        with torch.no_grad():
            va = run_epoch(model, val_loader, optimizer, device, cfg.model,
                           num_classes, cfg.ignore_index, criterion, train=False, scaler=None)
        scheduler.step()

        row = {
            "epoch":              epoch,
            "train_loss":         tr["loss"],
            "train_pixel_acc":    tr["pixel_acc"],
            "train_miou":         tr["miou"],
            "train_per_class_iou":tr["per_class_iou"],
            "val_loss":           va["loss"],
            "val_pixel_acc":      va["pixel_acc"],
            "val_miou":           va["miou"],
            "val_per_class_iou":  va["per_class_iou"],
            "lr":                 float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        with open(os.path.join(cfg.run_dir, "history.json"), "w") as f:
            json.dump({"history": history}, f, indent=2)

        # Print compact summary (without per-class lists to keep console readable)
        compact = {k: v for k, v in row.items() if "per_class" not in k}
        print(json.dumps(compact, indent=2))

        if va["miou"] > best_val_miou:
            best_val_miou = va["miou"]
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch, "val_miou": best_val_miou},
                best_path,
            )

    # Final test evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        te = run_epoch(model, test_loader, optimizer, device, cfg.model,
                       num_classes, cfg.ignore_index, criterion, train=False, scaler=None)

    summary = {
        "model":            cfg.model,
        "pretrained":       cfg.pretrained,
        "freeze_backbone":  cfg.freeze_backbone,
        "use_class_weights":cfg.use_class_weights,
        "use_amp":          cfg.use_amp,
        "num_classes":      num_classes,
        "best_val_miou":    float(best_val_miou),
        "test_miou":        float(te["miou"]),
        "test_pixel_acc":   float(te["pixel_acc"]),
        "test_loss":        float(te["loss"]),
        "test_per_class_iou": te["per_class_iou"],
        "device":           device.type,
        "elapsed_sec":      float(time.time() - t0),
    }
    with open(os.path.join(cfg.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps({k: v for k, v in summary.items() if k != "test_per_class_iou"}, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=str, default=os.path.join("asgn3_artifacts", "step1"))
    p.add_argument("--run_dir",       type=str, default=os.path.join("asgn3_runs", "debug_seg"))
    p.add_argument("--model", type=str, default="deeplabv3_resnet50",
                   choices=["custom_unet", "deeplabv3_resnet50", "deeplabv3_resnet101",
                            "fcn_resnet50", "segformer_b0"])
    p.add_argument("--pretrained",        action="store_true")
    p.add_argument("--freeze_backbone",   action="store_true")
    p.add_argument("--image_size",  type=int,   default=384)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--ignore_index",type=int,   default=DEFAULT_IGNORE_INDEX)
    p.add_argument("--use_class_weights", action="store_true",
                   help="Weight loss inversely by class pixel frequency (helps with imbalance)")
    p.add_argument("--use_amp",           action="store_true",
                   help="Enable mixed-precision training (torch.cuda.amp)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = SegTrainConfig(
        artifacts_dir    = args.artifacts_dir,
        run_dir          = args.run_dir,
        model            = args.model,
        pretrained       = bool(args.pretrained),
        freeze_backbone  = bool(args.freeze_backbone),
        image_size       = args.image_size,
        batch_size       = args.batch_size,
        epochs           = args.epochs,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        num_workers      = args.num_workers,
        seed             = args.seed,
        ignore_index     = args.ignore_index,
        use_class_weights= bool(args.use_class_weights),
        use_amp          = bool(args.use_amp),
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
