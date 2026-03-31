"""
Assignment 2 - Modular training script (CNN + ViT).

This script consumes artifacts created by `asgn2_01_preprocess_augment.py`:
- train_split.csv / val_split.csv / test_split.csv

It builds DataLoaders using `build_dataloaders()` and trains a classifier for `price_class`.
Designed to support later experiment regimes:
  (a) custom model you implement (custom_cnn) trained from scratch
  (b) pre-existing architecture trained from scratch (e.g., resnet18/vit_b_16 pretrained=False)
  (c) pre-trained weights WITHOUT fine-tuning (freeze_backbone=True)
  (d) pre-trained weights WITH fine-tuning (freeze_backbone=False)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from asgn2_01_preprocess_augment import (
    PreprocessConfig,
    build_dataloaders,
)


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torchvision
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch/torchvision not installed. Install them to run training.\n"
        "Example (CPU): pip install torch torchvision\n"
        "Example (Apple Silicon): see pytorch.org for the recommended install."
    ) from e


@dataclass(frozen=True)
class TrainConfig:
    artifacts_dir: str
    csv_path: str
    image_dir: str

    model: str
    num_classes: int = 3

    # Transfer learning toggles
    pretrained: bool = False
    freeze_backbone: bool = False

    # Training params
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42

    # Output
    run_dir: str = os.path.join("asgn2_runs", "debug_run")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """
    A simple CNN "model you developed yourselves" for regime (a).
    Kept intentionally small and standard for easy debugging.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _freeze_backbone_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_classifier_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def build_model(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "custom_cnn":
        # Regime (a): your own architecture, trained from scratch.
        return SmallCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)

        if freeze_backbone:
            _freeze_backbone_params(m)

        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        # Ensure classifier is trainable even when backbone is frozen.
        _unfreeze_classifier_params(m.fc)
        return m

    if model_name in ("vit_b_16", "vitb16", "vit"):
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
        m = torchvision.models.vit_b_16(weights=weights)

        if freeze_backbone:
            _freeze_backbone_params(m)

        # torchvision ViT exposes the classifier head as m.heads
        if hasattr(m, "heads") and hasattr(m.heads, "head"):
            in_features = m.heads.head.in_features
            m.heads.head = nn.Linear(in_features, num_classes)
            _unfreeze_classifier_params(m.heads.head)
        else:
            # Fallback for potential API changes.
            raise RuntimeError("Unexpected ViT classifier structure in torchvision.")

        return m

    raise ValueError(f"Unknown model '{model_name}'. Use one of: custom_cnn, resnet18, vit_b_16")


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += compute_accuracy(logits, y)
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "acc": total_acc / max(n_batches, 1)}


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        total_acc += compute_accuracy(logits, y)
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "acc": total_acc / max(n_batches, 1)}


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_splits(artifacts_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(artifacts_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(artifacts_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(artifacts_dir, "test_split.csv"))
    return train_df, val_df, test_df


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    os.makedirs(cfg.run_dir, exist_ok=True)
    save_json(os.path.join(cfg.run_dir, "train_config.json"), cfg.__dict__)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_splits(cfg.artifacts_dir)

    # Build loaders via step-1 module.
    preprocess_cfg = PreprocessConfig(
        csv_path=cfg.csv_path,
        image_dir=cfg.image_dir,
        output_dir=cfg.artifacts_dir,  # not used in loader building
        num_classes=cfg.num_classes,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=preprocess_cfg,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        model_name=cfg.model,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = -1.0
    best_path = os.path.join(cfg.run_dir, "best_model.pt")

    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_one_epoch(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        save_json(os.path.join(cfg.run_dir, "history.json"), {"history": history})

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc}, best_path)

        print(json.dumps(row, indent=2))

    # Final test evaluation using best checkpoint.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = eval_one_epoch(model, test_loader, device)

    summary = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_metrics["acc"]),
        "test_loss": float(test_metrics["loss"]),
        "device": device.type,
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(os.path.join(cfg.run_dir, "summary.json"), summary)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=str, default=os.path.join("asgn2_artifacts", "step3"))
    p.add_argument("--csv_path", type=str, default="socal2.csv")
    p.add_argument("--image_dir", type=str, default=os.path.join("socal2", "socal_pics"))

    p.add_argument("--model", type=str, default="resnet18", choices=["custom_cnn", "resnet18", "vit_b_16"])
    p.add_argument("--num_classes", type=int, default=3)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--run_dir", type=str, default=os.path.join("asgn2_runs", "debug_run"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        artifacts_dir=args.artifacts_dir,
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        model=args.model,
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained),
        freeze_backbone=bool(args.freeze_backbone),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

