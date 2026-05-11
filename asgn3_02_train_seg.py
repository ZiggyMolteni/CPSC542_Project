"""
Assignment 3 - Step 2  (v2 – improved pipeline)
Train semantic segmentation models with reusable, modular pipeline.

New in v2
---------
* Loss functions: CE (default), Dice+CE, Focal, Focal+Dice  (--loss_fn)
* Automatic class-frequency weighting               (--use_class_weights)
* Mixed-precision (AMP) training                    (--use_amp)
* Test-time augmentation: H-flip + multi-scale      (--use_tta / --tta_scales)
* Weighted random sampler for rare-class oversampling (--use_oversampling)
* Copy-paste augmentation for rare classes           (--use_copypaste)
* Larger ASPP dilations on DeepLab models            (--larger_aspp)
* deeplabv3_resnet101 backbone                       (--model deeplabv3_resnet101)
* CRF post-processing at eval time                  (--use_crf, requires pydensecrf)
* Richer training augmentations (rotation, colour-jitter, Gaussian blur)

Consumes artifacts from Step 1:
  asgn3_artifacts/step1/{train,val,test}_split.csv

Models:
  - custom_unet             (regime (a) - from scratch)
  - deeplabv3_resnet50      (CNN-based, pretrained ImageNet)
  - deeplabv3_resnet101     (CNN-based, pretrained ImageNet - stronger backbone)
  - fcn_resnet50            (CNN-based)
  - segformer_b0            (Transformer-based, requires `transformers`)

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
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    import torchvision
    import torchvision.transforms.functional as TF
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch/torchvision is required. Install with `pip install -r requirements.txt`."
    ) from e


DEFAULT_IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegTrainConfig:
    # Paths
    artifacts_dir: str = os.path.join("asgn3_artifacts", "step1")
    run_dir: str = os.path.join("asgn3_runs", "debug_seg")

    # Model
    model: str = "deeplabv3_resnet50"
    pretrained: bool = False
    freeze_backbone: bool = False

    # Training hyper-params
    image_size: int = 384
    batch_size: int = 4
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    ignore_index: int = DEFAULT_IGNORE_INDEX

    # v2 - loss & class weighting
    loss_fn: str = "ce"          # ce | dice_ce | focal | focal_dice
    use_class_weights: bool = False
    focal_gamma: float = 2.0

    # v2 - mixed precision
    use_amp: bool = False

    # v2 - test-time augmentation
    use_tta: bool = False
    tta_scales: str = "1.0"      # comma-separated float scales, e.g. "0.75,1.0,1.25"

    # v2 - rare-class handling
    use_oversampling: bool = False
    use_copypaste: bool = False
    copypaste_prob: float = 0.3
    rare_class_percentile: float = 30.0  # classes below this freq-percentile are "rare"

    # v2 - model architecture
    larger_aspp: bool = False    # replace ASPP dilations with larger values

    # v2 - CRF post-processing
    use_crf: bool = False
    crf_iters: int = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            raise ValueError(f"Raw class id out of [0,255]: {raw_id}")
        lut[raw_id] = new_id
    return lut


# ---------------------------------------------------------------------------
# Class statistics helpers
# ---------------------------------------------------------------------------

def compute_class_pixel_counts(
    df: pd.DataFrame, lut: np.ndarray, num_classes: int
) -> np.ndarray:
    """Count total pixels per class across all images in df."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, row in df.iterrows():
        m = lut[np.array(Image.open(row["mask_path"]), dtype=np.uint8)]
        for c in range(num_classes):
            counts[c] += int((m == c).sum())
    return counts


def identify_rare_classes(counts: np.ndarray, percentile: float) -> List[int]:
    """Return class indices whose pixel count is below *percentile*-th percentile."""
    present = counts[counts > 0]
    if len(present) == 0:
        return []
    threshold = np.percentile(present, percentile)
    return [c for c in range(len(counts)) if 0 < counts[c] <= threshold]


def compute_class_weights(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so mean weight equals 1."""
    n = len(counts)
    w = np.where(counts > 0, counts.sum() / (n * counts + 1e-8), 0.0)
    w = w / (w.sum() / n + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


def compute_sample_weights(
    df: pd.DataFrame, lut: np.ndarray, rare_ids: List[int]
) -> torch.Tensor:
    """Weight each sample by how much rare-class content it contains."""
    rare_set = set(rare_ids)
    weights = []
    for _, row in df.iterrows():
        m = lut[np.array(Image.open(row["mask_path"]), dtype=np.uint8)]
        rare_pixels = sum(int((m == c).sum()) for c in rare_set)
        weights.append(1.0 + rare_pixels / max(m.size, 1))
    return torch.tensor(weights, dtype=torch.double)


# ---------------------------------------------------------------------------
# Copy-paste augmentation
# ---------------------------------------------------------------------------

class CopyPasteAugmentation:
    """
    Paste foreground objects of rare classes from a pool of source images
    onto the current training image (applied after resize, before normalise).

    img_np  : H x W x 3  uint8
    mask_np : H x W      int64 (remapped contiguous IDs)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lut: np.ndarray,
        image_size: int,
        rare_ids: List[int],
        p: float = 0.3,
    ):
        self.df = df.reset_index(drop=True)
        self.lut = lut
        self.image_size = image_size
        self.rare_set = set(rare_ids)
        self.p = p
        # Pre-scan which images contain rare classes
        self._rare_indices: List[int] = []
        for i, row in self.df.iterrows():
            m = lut[np.array(Image.open(row["mask_path"]), dtype=np.uint8)]
            if any((m == c).any() for c in self.rare_set):
                self._rare_indices.append(int(i))

    def __call__(
        self, img_np: np.ndarray, mask_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self.p or not self._rare_indices:
            return img_np, mask_np

        src_idx = random.choice(self._rare_indices)
        src_row = self.df.iloc[src_idx]
        src_img = np.array(
            Image.open(src_row["image_path"])
            .convert("RGB")
            .resize((self.image_size, self.image_size), Image.BILINEAR)
        )
        src_mask = self.lut[
            np.array(
                Image.open(src_row["mask_path"]).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                ),
                dtype=np.uint8,
            )
        ]

        present = [c for c in self.rare_set if (src_mask == c).any()]
        if not present:
            return img_np, mask_np

        cls = random.choice(present)
        obj_mask = src_mask == cls

        result_img = img_np.copy()
        result_mask = mask_np.copy()
        result_img[obj_mask] = src_img[obj_mask]
        result_mask[obj_mask] = cls
        return result_img, result_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        lut: np.ndarray,
        ignore_index: int,
        train: bool,
        copypaste: Optional[CopyPasteAugmentation] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.lut = lut
        self.ignore_index = ignore_index
        self.train = train
        self.copypaste = copypaste

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img  = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"])

        # Resize (NEAREST for mask avoids label bleeding)
        img  = TF.resize(img,  [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            # --- Copy-paste (operates on numpy before PIL augmentations) ---
            img_np  = np.array(img, dtype=np.uint8)
            mask_np = self.lut[np.array(mask, dtype=np.uint8)].astype(np.int64)

            if self.copypaste is not None:
                img_np, mask_np = self.copypaste(img_np, mask_np)

            img  = Image.fromarray(img_np)
            # Keep mask as numpy for final conversion; skip second lut application below
            lut_already_applied = True

            # --- Geometric augmentations (joint on img+mask) ---
            if random.random() < 0.5:   # horizontal flip
                img  = TF.hflip(img)
                mask_np = mask_np[:, ::-1].copy()

            if random.random() < 0.2:   # vertical flip
                img  = TF.vflip(img)
                mask_np = mask_np[::-1, :].copy()

            if random.random() < 0.3:   # random rotation +-10 deg
                angle = random.uniform(-10, 10)
                img  = TF.rotate(img, angle,
                                  interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                mask_pil = TF.rotate(mask_pil, angle,
                                      interpolation=TF.InterpolationMode.NEAREST,
                                      fill=self.ignore_index)
                mask_np  = np.array(mask_pil, dtype=np.int64)

            if random.random() < 0.5:   # random crop + resize
                scale = random.uniform(0.75, 1.0)
                ch    = int(self.image_size * scale)
                cw    = int(self.image_size * scale)
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                    img, output_size=(ch, cw)
                )
                img     = TF.crop(img, i, j, h, w)
                mask_np = mask_np[i:i+h, j:j+w]
                img     = TF.resize(img, [self.image_size, self.image_size],
                                    interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                mask_pil = TF.resize(mask_pil, [self.image_size, self.image_size],
                                     interpolation=TF.InterpolationMode.NEAREST)
                mask_np  = np.array(mask_pil, dtype=np.int64)

            # --- Photometric augmentations (image only) ---
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        else:
            lut_already_applied = False
            mask_np = None

        # --- To tensor + normalise ---
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if lut_already_applied and mask_np is not None:
            y = torch.from_numpy(mask_np.astype(np.int64))
        else:
            m = self.lut[np.array(mask, dtype=np.uint8)].astype(np.int64)
            y = torch.from_numpy(m)

        return x, y


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss averaged over classes."""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)          # B x C x H x W
        valid = (targets != self.ignore_index)    # B x H x W

        tgt = targets.clone()
        tgt[~valid] = 0                           # zero-out ignored pixels (safe for one_hot)

        one_hot = (
            F.one_hot(tgt, self.num_classes)      # B x H x W x C
            .permute(0, 3, 1, 2)                  # B x C x H x W
            .float()
        )
        mask    = valid.unsqueeze(1).float()
        probs   = probs   * mask
        one_hot = one_hot * mask

        dims        = (0, 2, 3)
        inter       = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)
        dice_score  = (2.0 * inter + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weight tensor."""

    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = 255,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, l1: nn.Module, l2: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.l1    = l1
        self.l2    = l2
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.l1(logits, targets) + (1.0 - self.alpha) * self.l2(logits, targets)


def build_criterion(
    loss_fn: str,
    num_classes: int,
    ignore_index: int,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    fn    = loss_fn.lower().strip()
    ce    = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    dice  = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    focal = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index, weight=class_weights)

    if fn == "ce":
        return ce
    if fn == "dice_ce":
        return CombinedLoss(ce, dice, alpha=0.5)
    if fn == "focal":
        return focal
    if fn == "focal_dice":
        return CombinedLoss(focal, dice, alpha=0.5)
    raise ValueError(
        f"Unknown loss_fn '{loss_fn}'. Choices: ce, dice_ce, focal, focal_dice"
    )


# ---------------------------------------------------------------------------
# Model architectures
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
            dh   = skip.shape[-2] - x.shape[-2]
            dw   = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2: dh // 2 + x.shape[-2],
                               dw // 2: dw // 2 + x.shape[-1]]
        return self.conv(torch.cat([skip, x], dim=1))


class CustomUNet(nn.Module):
    """Small U-Net for semantic segmentation (regime (a) - trained from scratch)."""

    def __init__(self, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.enc1       = DoubleConv(3, base_ch)
        self.enc2       = DoubleConv(base_ch,     base_ch * 2)
        self.enc3       = DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool       = nn.MaxPool2d(2)
        self.up1        = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2        = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3        = Up(base_ch * 2, base_ch,     base_ch)
        self.outc       = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x  = self.bottleneck(self.pool(s3))
        x  = self.up1(x, s3)
        x  = self.up2(x, s2)
        x  = self.up3(x, s1)
        return self.outc(x)


def _patch_aspp_dilations(
    model: nn.Module, new_dilations: Tuple[int, ...] = (18, 36, 54)
) -> nn.Module:
    """
    Replace the dilated convolutions inside a DeepLabV3 ASPP head with larger
    dilations to capture more global context.  Falls back silently on failure.
    """
    try:
        aspp  = model.classifier[0]   # torchvision ASPP module
        convs = list(aspp.convs)      # [1x1conv, ASPPConv x3, ASPPPooling]
        d_idx = 0
        for i, module in enumerate(convs):
            children = list(module.children())
            if not children:
                continue
            first = children[0]
            # ASPPConv has a dilated Conv2d as its first child
            if (
                isinstance(first, nn.Conv2d)
                and first.dilation not in {(1, 1), (0, 0)}
                and d_idx < len(new_dilations)
            ):
                d = new_dilations[d_idx]
                convs[i] = nn.Sequential(
                    nn.Conv2d(first.in_channels, first.out_channels,
                              3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(first.out_channels),
                    nn.ReLU(inplace=True),
                )
                d_idx += 1
        aspp.convs = nn.ModuleList(convs)
        print(f"[ASPP] Patched {d_idx} dilated convs -> {new_dilations[:d_idx]}")
    except Exception as exc:
        print(f"[ASPP] Could not patch dilations ({exc}); using model defaults.")
    return model


def build_seg_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    larger_aspp: bool = False,
) -> nn.Module:
    name = model_name.lower().strip()

    # ---- custom U-Net ----
    if name == "custom_unet":
        if pretrained:
            raise ValueError("custom_unet does not support pretrained weights.")
        m = CustomUNet(num_classes=num_classes)
        if freeze_backbone:
            for block in [m.enc1, m.enc2, m.enc3, m.bottleneck]:
                for p in block.parameters():
                    p.requires_grad = False
        return m

    # ---- DeepLabV3 ResNet-50 ----
    if name == "deeplabv3_resnet50":
        if pretrained:
            w = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=w)
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, num_classes, 1)
            if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
                aux_in = m.aux_classifier[-1].in_channels
                m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, 1)
        else:
            m = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None, num_classes=num_classes
            )
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        if larger_aspp:
            m = _patch_aspp_dilations(m, (18, 36, 54))
        return m

    # ---- DeepLabV3 ResNet-101  (v2 addition) ----
    if name == "deeplabv3_resnet101":
        if pretrained:
            w = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
            m = torchvision.models.segmentation.deeplabv3_resnet101(weights=w)
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, num_classes, 1)
            if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
                aux_in = m.aux_classifier[-1].in_channels
                m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, 1)
        else:
            m = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=None, num_classes=num_classes
            )
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        if larger_aspp:
            m = _patch_aspp_dilations(m, (18, 36, 54))
        return m

    # ---- FCN ResNet-50 ----
    if name == "fcn_resnet50":
        if pretrained:
            w = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
            m = torchvision.models.segmentation.fcn_resnet50(weights=w)
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, num_classes, 1)
            if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
                aux_in = m.aux_classifier[-1].in_channels
                m.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, 1)
        else:
            m = torchvision.models.segmentation.fcn_resnet50(
                weights=None, num_classes=num_classes
            )
        if freeze_backbone:
            for p in m.backbone.parameters():
                p.requires_grad = False
        return m

    # ---- SegFormer-B0 ----
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
        "Choices: custom_unet, deeplabv3_resnet50, deeplabv3_resnet101, "
        "fcn_resnet50, segformer_b0"
    )


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

_TORCHVISION_DICT_MODELS = {
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "fcn_resnet50",
}


def forward_logits(
    model: nn.Module, x: torch.Tensor, model_name: str
) -> torch.Tensor:
    name = model_name.lower().strip()
    out  = model(x)

    if name in _TORCHVISION_DICT_MODELS:
        return out["out"]

    if name == "custom_unet":
        return out

    if name == "segformer_b0":
        return out.logits

    raise ValueError(f"Unsupported model_name in forward_logits: {model_name}")


@torch.no_grad()
def tta_forward_logits(
    model: nn.Module,
    x: torch.Tensor,
    model_name: str,
    scales: List[float],
    hflip: bool = True,
) -> torch.Tensor:
    """
    Test-time augmentation: average softmax probabilities over multiple scales
    and their horizontal flips.  Returns averaged probabilities (not raw logits).
    The caller should use argmax for predictions but NOT pass the result to a
    standard cross-entropy loss (use loss from the plain forward pass instead).
    """
    _, _, H, W = x.shape
    prob_sum: Optional[torch.Tensor] = None
    count = 0

    for scale in scales:
        for do_flip in ([False, True] if hflip else [False]):
            xi = x
            if do_flip:
                xi = torch.flip(xi, dims=[3])
            if abs(scale - 1.0) > 1e-3:
                new_h = max(1, int(H * scale))
                new_w = max(1, int(W * scale))
                xi    = F.interpolate(xi, size=(new_h, new_w),
                                      mode="bilinear", align_corners=False)

            logits = forward_logits(model, xi, model_name)
            logits = F.interpolate(logits, size=(H, W),
                                   mode="bilinear", align_corners=False)
            if do_flip:
                logits = torch.flip(logits, dims=[3])

            probs = F.softmax(logits, dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            count += 1

    return prob_sum / count


# ---------------------------------------------------------------------------
# CRF post-processing
# ---------------------------------------------------------------------------

def apply_crf(
    image_np: np.ndarray,
    probs_np: np.ndarray,
    n_iters: int = 5,
) -> np.ndarray:
    """
    Dense CRF refinement.

    Parameters
    ----------
    image_np  : H x W x 3  uint8
    probs_np  : C x H x W  float32  (softmax probabilities)

    Returns
    -------
    H x W int64 predicted class map
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("[CRF] pydensecrf not installed; skipping. pip install pydensecrf")
        return probs_np.argmax(axis=0).astype(np.int64)

    C, H, W = probs_np.shape
    d = dcrf.DenseCRF2D(W, H, C)

    U = unary_from_softmax(probs_np)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(
        sxy=80, srgb=13,
        rgbim=np.ascontiguousarray(image_np),
        compat=10,
    )

    Q = d.inference(n_iters)
    return np.array(Q, dtype=np.float32).reshape(C, H, W).argmax(axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_batch_metrics(
    logits_or_probs: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, float]:
    """Works with either raw logits or averaged softmax probabilities."""
    preds = torch.argmax(logits_or_probs, dim=1)
    valid = y != ignore_index

    if valid.sum().item() == 0:
        return {"pixel_acc": 0.0, "miou": 0.0}

    pixel_acc = float((preds[valid] == y[valid]).float().mean().item())

    ious = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        y_c    = (y    == c) & valid
        inter  = (pred_c & y_c).sum().item()
        union  = (pred_c | y_c).sum().item()
        if union > 0:
            ious.append(inter / union)

    miou = float(np.mean(ious)) if ious else 0.0
    return {"pixel_acc": pixel_acc, "miou": miou}


def sanity_check_batch(loader, num_classes: int, ignore_index: int) -> None:
    x, y = next(iter(loader))
    assert x.ndim == 4 and y.ndim == 3, f"Shape error: x={x.shape}, y={y.shape}"
    assert x.shape[0] == y.shape[0],    "Batch-size mismatch image/mask"
    assert x.shape[2:] == y.shape[1:],  "Spatial mismatch image/mask"
    y_np = y.numpy()
    bad  = ((y_np < 0) | ((y_np >= num_classes) & (y_np != ignore_index))).any()
    if bad:
        u = np.unique(y_np)
        raise ValueError(
            f"Mask has unexpected IDs: {u.tolist()}, "
            f"expected [0..{num_classes-1}] + ignore_index={ignore_index}"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    model_name: str,
    num_classes: int,
    ignore_index: int,
    criterion: nn.Module,
    train: bool,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    use_tta: bool = False,
    tta_scales: Optional[List[float]] = None,
) -> Dict[str, float]:
    model.train() if train else model.eval()

    if tta_scales is None:
        tta_scales = [1.0]

    total_loss = total_acc = total_miou = 0.0
    n = 0

    amp_enabled = use_amp and device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                logits = forward_logits(model, x, model_name)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=y.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                loss = criterion(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # Metrics: use TTA during evaluation if requested
            with torch.no_grad():
                if (not train) and use_tta:
                    probs = tta_forward_logits(
                        model, x, model_name, scales=tta_scales, hflip=True
                    )
                    metrics = compute_batch_metrics(
                        probs, y, num_classes=num_classes, ignore_index=ignore_index
                    )
                else:
                    metrics = compute_batch_metrics(
                        logits, y, num_classes=num_classes, ignore_index=ignore_index
                    )

            total_loss += float(loss.item())
            total_acc  += metrics["pixel_acc"]
            total_miou += metrics["miou"]
            n += 1

    return {
        "loss":      total_loss / max(n, 1),
        "pixel_acc": total_acc  / max(n, 1),
        "miou":      total_miou / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def run_training(cfg: SegTrainConfig) -> Dict[str, float]:
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- Data splits ---
    train_df, val_df, test_df = load_splits(cfg.artifacts_dir)

    class_ids   = infer_class_ids_from_masks(
        [train_df, val_df, test_df], ignore_index=cfg.ignore_index
    )
    num_classes = len(class_ids)
    lut         = build_lookup_table(class_ids, ignore_index=cfg.ignore_index)

    with open(os.path.join(cfg.run_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {
                "raw_class_ids":  class_ids,
                "remapped_to":    {str(r): i for i, r in enumerate(class_ids)},
                "num_classes":    num_classes,
                "ignore_index":   cfg.ignore_index,
            },
            f, indent=2,
        )

    # --- Class statistics ---
    print("[stats] Computing class pixel counts ...")
    counts   = compute_class_pixel_counts(train_df, lut, num_classes)
    rare_ids = identify_rare_classes(counts, percentile=cfg.rare_class_percentile)
    print(f"[stats] Rare classes (bottom {cfg.rare_class_percentile}th percentile): {rare_ids}")

    # --- Class weights ---
    class_weights: Optional[torch.Tensor] = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(counts).to(device)
        print(f"[loss] Class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    # --- Loss criterion ---
    criterion = build_criterion(
        cfg.loss_fn,
        num_classes=num_classes,
        ignore_index=cfg.ignore_index,
        focal_gamma=cfg.focal_gamma,
        class_weights=class_weights,
    ).to(device)
    print(f"[loss] Using '{cfg.loss_fn}'")

    # --- Copy-paste augmentation ---
    copypaste: Optional[CopyPasteAugmentation] = None
    if cfg.use_copypaste and rare_ids:
        print(f"[aug] Setting up copy-paste for rare classes {rare_ids} ...")
        copypaste = CopyPasteAugmentation(
            df=train_df, lut=lut,
            image_size=cfg.image_size,
            rare_ids=rare_ids,
            p=cfg.copypaste_prob,
        )
    elif cfg.use_copypaste:
        print("[aug] No rare classes identified; copy-paste disabled.")

    # --- Datasets ---
    train_ds = SegmentationDataset(
        train_df, image_size=cfg.image_size, lut=lut,
        ignore_index=cfg.ignore_index, train=True, copypaste=copypaste,
    )
    val_ds = SegmentationDataset(
        val_df,   image_size=cfg.image_size, lut=lut,
        ignore_index=cfg.ignore_index, train=False,
    )
    test_ds = SegmentationDataset(
        test_df,  image_size=cfg.image_size, lut=lut,
        ignore_index=cfg.ignore_index, train=False,
    )

    # --- Sampler (oversampling) ---
    train_sampler = None
    train_shuffle = True
    if cfg.use_oversampling and rare_ids:
        print(f"[sampler] Building WeightedRandomSampler for rare classes {rare_ids} ...")
        sample_w      = compute_sample_weights(train_df, lut, rare_ids)
        train_sampler = WeightedRandomSampler(
            weights=sample_w, num_samples=len(sample_w), replacement=True
        )
        train_shuffle = False   # sampler and shuffle are mutually exclusive

    # --- Data loaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    sanity_check_batch(train_loader, num_classes=num_classes, ignore_index=cfg.ignore_index)

    # --- Model ---
    model = build_seg_model(
        cfg.model,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        larger_aspp=cfg.larger_aspp,
    ).to(device)

    # --- Optimiser + scheduler ---
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    # --- AMP scaler ---
    scaler: Optional[GradScaler] = None
    if cfg.use_amp:
        if device.type == "cuda":
            scaler = GradScaler()
            print("[amp] Mixed precision enabled")
        else:
            print("[amp] AMP requested but device is CPU; skipping.")

    # --- TTA scales ---
    tta_scales = [float(s.strip()) for s in cfg.tta_scales.split(",") if s.strip()]

    # --- Training loop ---
    best_val_miou = -1.0
    best_path     = os.path.join(cfg.run_dir, "best_model.pt")
    history: List[dict] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, device, cfg.model,
            num_classes, cfg.ignore_index, criterion,
            train=True, use_amp=cfg.use_amp, scaler=scaler,
        )
        va = run_epoch(
            model, val_loader, None, device, cfg.model,
            num_classes, cfg.ignore_index, criterion,
            train=False,
            use_tta=cfg.use_tta, tta_scales=tta_scales,
        )
        scheduler.step()

        row = {
            "epoch":           epoch,
            "train_loss":      tr["loss"],
            "train_pixel_acc": tr["pixel_acc"],
            "train_miou":      tr["miou"],
            "val_loss":        va["loss"],
            "val_pixel_acc":   va["pixel_acc"],
            "val_miou":        va["miou"],
            "lr":              float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        with open(os.path.join(cfg.run_dir, "history.json"), "w") as f:
            json.dump({"history": history}, f, indent=2)
        print(json.dumps(row, indent=2))

        if va["miou"] > best_val_miou:
            best_val_miou = va["miou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch,
                    "val_miou":         best_val_miou,
                },
                best_path,
            )

    # --- Test evaluation ---
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    te = run_epoch(
        model, test_loader, None, device, cfg.model,
        num_classes, cfg.ignore_index, criterion,
        train=False,
        use_tta=cfg.use_tta, tta_scales=tta_scales,
    )

    summary = {
        "model":              cfg.model,
        "pretrained":         cfg.pretrained,
        "freeze_backbone":    cfg.freeze_backbone,
        "loss_fn":            cfg.loss_fn,
        "use_class_weights":  cfg.use_class_weights,
        "use_amp":            cfg.use_amp,
        "use_tta":            cfg.use_tta,
        "tta_scales":         tta_scales,
        "use_oversampling":   cfg.use_oversampling,
        "use_copypaste":      cfg.use_copypaste,
        "larger_aspp":        cfg.larger_aspp,
        "use_crf":            cfg.use_crf,
        "num_classes":        num_classes,
        "best_val_miou":      float(best_val_miou),
        "test_miou":          float(te["miou"]),
        "test_pixel_acc":     float(te["pixel_acc"]),
        "test_loss":          float(te["loss"]),
        "device":             device.type,
        "elapsed_sec":        float(time.time() - t0),
    }
    with open(os.path.join(cfg.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train semantic segmentation – Assignment 3 v2"
    )
    # Paths
    p.add_argument("--artifacts_dir", default=os.path.join("asgn3_artifacts", "step1"))
    p.add_argument("--run_dir",       default=os.path.join("asgn3_runs", "debug_seg"))

    # Model
    p.add_argument(
        "--model", default="deeplabv3_resnet50",
        choices=[
            "custom_unet", "deeplabv3_resnet50", "deeplabv3_resnet101",
            "fcn_resnet50", "segformer_b0",
        ],
    )
    p.add_argument("--pretrained",      action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument(
        "--larger_aspp", action="store_true",
        help="Replace default ASPP dilations [12,24,36] with [18,36,54]",
    )

    # Training hyper-params
    p.add_argument("--image_size",   type=int,   default=384)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--ignore_index", type=int,   default=DEFAULT_IGNORE_INDEX)

    # Loss
    p.add_argument(
        "--loss_fn", default="ce",
        choices=["ce", "dice_ce", "focal", "focal_dice"],
        help="Loss function (default: ce)",
    )
    p.add_argument(
        "--use_class_weights", action="store_true",
        help="Weight loss by inverse class pixel frequency",
    )
    p.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="Gamma for focal loss (default: 2.0)",
    )

    # Mixed precision
    p.add_argument(
        "--use_amp", action="store_true",
        help="Enable automatic mixed precision (AMP) training (CUDA only)",
    )

    # Test-time augmentation
    p.add_argument(
        "--use_tta", action="store_true",
        help="Enable test-time augmentation (H-flip + multi-scale) during evaluation",
    )
    p.add_argument(
        "--tta_scales", default="1.0",
        help="Comma-separated scale factors for TTA (e.g. '0.75,1.0,1.25')",
    )

    # Rare-class handling
    p.add_argument(
        "--use_oversampling", action="store_true",
        help="Use WeightedRandomSampler to oversample images with rare classes",
    )
    p.add_argument(
        "--use_copypaste", action="store_true",
        help="Copy-paste rare-class segments into training images",
    )
    p.add_argument(
        "--copypaste_prob", type=float, default=0.3,
        help="Probability of applying copy-paste per training image (default: 0.3)",
    )
    p.add_argument(
        "--rare_class_percentile", type=float, default=30.0,
        help="Classes below this pixel-count percentile are treated as rare (default: 30)",
    )

    # CRF
    p.add_argument(
        "--use_crf", action="store_true",
        help="Apply Dense CRF post-processing at test time (requires pydensecrf)",
    )
    p.add_argument(
        "--crf_iters", type=int, default=5,
        help="Number of CRF inference iterations (default: 5)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = SegTrainConfig(
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
        loss_fn=args.loss_fn,
        use_class_weights=bool(args.use_class_weights),
        focal_gamma=args.focal_gamma,
        use_amp=bool(args.use_amp),
        use_tta=bool(args.use_tta),
        tta_scales=args.tta_scales,
        use_oversampling=bool(args.use_oversampling),
        use_copypaste=bool(args.use_copypaste),
        copypaste_prob=args.copypaste_prob,
        rare_class_percentile=args.rare_class_percentile,
        larger_aspp=bool(args.larger_aspp),
        use_crf=bool(args.use_crf),
        crf_iters=args.crf_iters,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
