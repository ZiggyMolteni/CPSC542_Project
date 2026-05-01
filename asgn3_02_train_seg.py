"""
Assignment 3 - Step 2
Train semantic segmentation models with reusable, modular pipeline.

Consumes artifacts from Step 1:
  asgn3_artifacts/step1/{train,val,test}_split.csv

Models:
  - custom_unet         (your own small U-Net from scratch; regime (a))
  - deeplabv3_resnet50  (CNN-based)
  - fcn_resnet50        (CNN-based)
  - segformer_b0        (Transformer-based, requires `transformers`)

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
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(artifacts_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(artifacts_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(artifacts_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv"))
    return train_df, val_df, test_df


def infer_class_ids_from_masks(
    dfs: List[pd.DataFrame],
    ignore_index: int,
) -> List[int]:
    class_ids = set()
    for df in dfs:
        for p in df["mask_path"].tolist():
            m = np.array(Image.open(p))
            ids = np.unique(m).tolist()
            for i in ids:
                i = int(i)
                if i == ignore_index:
                    continue
                class_ids.add(i)
    return sorted(class_ids)


def build_lookup_table(class_ids: List[int], ignore_index: int) -> np.ndarray:
    """
    Maps raw mask IDs -> contiguous [0..K-1], unknown -> ignore_index.
    Assumes masks are uint8 (CMP uses 1..12 style IDs).
    """
    lut = np.full(256, ignore_index, dtype=np.uint8)
    for new_id, raw_id in enumerate(class_ids):
        if raw_id < 0 or raw_id > 255:
            raise ValueError(f"Raw class id out of [0,255] range: {raw_id}")
        lut[raw_id] = new_id
    return lut


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
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"])

        # Resize first (keeps image/mask aligned). Nearest for masks.
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        m = np.array(mask, dtype=np.uint8)
        m = self.lut[m]  # remap raw ids to contiguous ids
        y = torch.from_numpy(m.astype(np.int64))
        return x, y


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
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd sizes by center-cropping skip to match x.
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2 : dh // 2 + x.shape[-2], dw // 2 : dw // 2 + x.shape[-1]]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CustomUNet(nn.Module):
    """
    Small U-Net for semantic segmentation (regime (a)).
    Intended to be trained from scratch on modest datasets.
    """

    def __init__(self, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(3, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        return self.outc(x)


def build_seg_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "custom_unet":
        if pretrained:
            raise ValueError("custom_unet does not support pretrained weights (regime (a) is scratch-only).")
        m = CustomUNet(num_classes=num_classes)
        if freeze_backbone:
            # Freeze encoder-ish path for a rough analogue of "frozen backbone".
            for p in m.enc1.parameters():
                p.requires_grad = False
            for p in m.enc2.parameters():
                p.requires_grad = False
            for p in m.enc3.parameters():
                p.requires_grad = False
            for p in m.bottleneck.parameters():
                p.requires_grad = False
        return m

    if model_name == "deeplabv3_resnet50":
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        # torchvision enforces COCO class count when weights are provided.
        # So load pretrained model first, then replace classifier head.
        if pretrained:
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
            if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
                aux_in = m.aux_classifier[-1].in_channels
                m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
        else:
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    if model_name == "fcn_resnet50":
        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
        if pretrained:
            m = torchvision.models.segmentation.fcn_resnet50(weights=weights)
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
            if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
                aux_in = m.aux_classifier[-1].in_channels
                m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
        else:
            m = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    if model_name == "segformer_b0":
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "segformer_b0 requires `transformers`. Install with `pip install transformers`."
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

    raise ValueError("Unknown model. Use: custom_unet, deeplabv3_resnet50, fcn_resnet50, segformer_b0")


def forward_logits(model: nn.Module, x: torch.Tensor, model_name: str) -> torch.Tensor:
    name = model_name.lower().strip()
    out = model(x)

    # torchvision segmentation models return dict with key "out".
    if name in {"deeplabv3_resnet50", "fcn_resnet50"}:
        logits = out["out"]
        return logits

    if name == "custom_unet":
        return out

    # HF segformer returns object with .logits
    if name == "segformer_b0":
        logits = out.logits
        return logits

    raise ValueError(f"Unsupported model_name: {model_name}")


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
        return {"pixel_acc": 0.0, "miou": 0.0}

    # Pixel accuracy
    pixel_acc = float((preds[valid] == y[valid]).float().mean().item())

    # mIoU
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        y_c = (y == c) & valid
        inter = (pred_c & y_c).sum().item()
        union = (pred_c | y_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return {"pixel_acc": pixel_acc, "miou": miou}


def sanity_check_batch(loader, num_classes: int, ignore_index: int) -> None:
    x, y = next(iter(loader))
    assert x.ndim == 4 and y.ndim == 3, f"Unexpected shapes: x={x.shape}, y={y.shape}"
    assert x.shape[0] == y.shape[0], "Batch size mismatch between image and mask"
    assert x.shape[2:] == y.shape[1:], "Spatial mismatch between image and mask"

    y_np = y.numpy()
    # IDs should be [0..K-1] plus ignore index.
    bad = ((y_np < 0) | ((y_np >= num_classes) & (y_np != ignore_index))).any()
    if bad:
        u = np.unique(y_np)
        raise ValueError(
            f"Mask contains unexpected IDs after remap. Unique IDs: {u.tolist()}, "
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
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = forward_logits(model, x, model_name=model_name)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        metrics = compute_batch_metrics(logits, y, num_classes=num_classes, ignore_index=ignore_index)
        total_loss += float(loss.item())
        total_acc += metrics["pixel_acc"]
        total_miou += metrics["miou"]
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "pixel_acc": total_acc / max(n, 1),
        "miou": total_miou / max(n, 1),
    }


def run_training(cfg: SegTrainConfig) -> Dict[str, float]:
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "train_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_splits(cfg.artifacts_dir)
    class_ids = infer_class_ids_from_masks([train_df, val_df, test_df], ignore_index=cfg.ignore_index)
    num_classes = len(class_ids)
    lut = build_lookup_table(class_ids, ignore_index=cfg.ignore_index)

    with open(os.path.join(cfg.run_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {
                "raw_class_ids": class_ids,
                "remapped_to": {str(raw): i for i, raw in enumerate(class_ids)},
                "num_classes": num_classes,
                "ignore_index": cfg.ignore_index,
            },
            f,
            indent=2,
        )

    train_ds = SegmentationDataset(
        train_df,
        image_size=cfg.image_size,
        lut=lut,
        ignore_index=cfg.ignore_index,
        train=True,
    )
    val_ds = SegmentationDataset(
        val_df,
        image_size=cfg.image_size,
        lut=lut,
        ignore_index=cfg.ignore_index,
        train=False,
    )
    test_ds = SegmentationDataset(
        test_df,
        image_size=cfg.image_size,
        lut=lut,
        ignore_index=cfg.ignore_index,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Sanity check before training.
    sanity_check_batch(train_loader, num_classes=num_classes, ignore_index=cfg.ignore_index)

    model = build_seg_model(
        cfg.model,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_miou = -1.0
    best_path = os.path.join(cfg.run_dir, "best_model.pt")
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, device, cfg.model, num_classes, cfg.ignore_index, train=True
        )
        with torch.no_grad():
            va = run_epoch(
                model, val_loader, optimizer, device, cfg.model, num_classes, cfg.ignore_index, train=False
            )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_pixel_acc": tr["pixel_acc"],
            "train_miou": tr["miou"],
            "val_loss": va["loss"],
            "val_pixel_acc": va["pixel_acc"],
            "val_miou": va["miou"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        with open(os.path.join(cfg.run_dir, "history.json"), "w") as f:
            json.dump({"history": history}, f, indent=2)
        print(json.dumps(row, indent=2))

        if va["miou"] > best_val_miou:
            best_val_miou = va["miou"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_miou": best_val_miou},
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        te = run_epoch(
            model, test_loader, optimizer, device, cfg.model, num_classes, cfg.ignore_index, train=False
        )

    summary = {
        "model": cfg.model,
        "pretrained": cfg.pretrained,
        "freeze_backbone": cfg.freeze_backbone,
        "num_classes": num_classes,
        "best_val_miou": float(best_val_miou),
        "test_miou": float(te["miou"]),
        "test_pixel_acc": float(te["pixel_acc"]),
        "test_loss": float(te["loss"]),
        "device": device.type,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(os.path.join(cfg.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=str, default=os.path.join("asgn3_artifacts", "step1"))
    p.add_argument("--run_dir", type=str, default=os.path.join("asgn3_runs", "debug_seg"))

    p.add_argument(
        "--model",
        type=str,
        default="deeplabv3_resnet50",
        choices=["custom_unet", "deeplabv3_resnet50", "fcn_resnet50", "segformer_b0"],
    )
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore_index", type=int, default=DEFAULT_IGNORE_INDEX)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegTrainConfig(
        artifacts_dir=args.artifacts_dir,
        run_dir=args.run_dir,
        model=args.model,
        pretrained=bool(args.pretrained),
        freeze_backbone=bool(args.freeze_backbone),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        ignore_index=args.ignore_index,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

