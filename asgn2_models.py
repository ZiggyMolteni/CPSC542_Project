"""
Shared CNN / ViT model factory for Assignment 2.
Used by `asgn2_02_train.py` and `asgn2_03_visualize.py` so viz does not import the trainer.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torchvision
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch/torchvision not installed. Install with:\n"
        "  pip install torch torchvision\n"
        "Or: pip install -r requirements.txt\n"
        "Apple Silicon: https://pytorch.org/get-started/locally/"
    ) from e


class SmallCNN(nn.Module):
    """
    A simple CNN for regime (a) — trained from scratch.
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
        return SmallCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)

        if freeze_backbone:
            _freeze_backbone_params(m)

        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        _unfreeze_classifier_params(m.fc)
        return m

    if model_name in ("vit_b_16", "vitb16", "vit"):
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
        m = torchvision.models.vit_b_16(weights=weights)

        if freeze_backbone:
            _freeze_backbone_params(m)

        if hasattr(m, "heads") and hasattr(m.heads, "head"):
            in_features = m.heads.head.in_features
            m.heads.head = nn.Linear(in_features, num_classes)
            _unfreeze_classifier_params(m.heads.head)
        else:
            raise RuntimeError("Unexpected ViT classifier structure in torchvision.")

        return m

    raise ValueError(f"Unknown model '{model_name}'. Use one of: custom_cnn, resnet18, vit_b_16")


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())
