"""

Outputs:   chart_training_curves.png   (upload this to Claude to embed in docx)
           confusion_matrix_cmp.png    (already downloaded above — upload to Claude)
           confusion_matrix_ade.png    (same)
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os

files = {
    "ResNet50 100ep": "history_resnet50_100ep.json",
    "ResNet50 200ep": "history_resnet50_200ep.json",
    "ResNet101 100ep ★": "history_resnet101.json",
}

missing = [k for k, v in files.items() if not os.path.exists(v)]
if missing:
    print(f"Missing files: {missing}")
    print("Run the scp commands at the top of this script first.")
    sys.exit(1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {"ResNet50 100ep": "#90CAF9", "ResNet50 200ep": "#42A5F5", "ResNet101 100ep ★": "#1565C0"}
styles = {"ResNet50 100ep": "--", "ResNet50 200ep": "-.", "ResNet101 100ep ★": "-"}

for label, fname in files.items():
    with open(fname) as f:
        data = json.load(f)
    history = data["history"]
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_miou = [r["val_miou"] for r in history]

    axes[0].plot(epochs, train_loss, label=label, color=colors[label],
                 linestyle=styles[label], linewidth=1.8)
    axes[1].plot(epochs, val_miou, label=label, color=colors[label],
                 linestyle=styles[label], linewidth=1.8)

for ax, title, ylabel in zip(
    axes,
    ["Training Loss per Epoch", "Validation mIoU per Epoch"],
    ["Cross-Entropy Loss", "Validation mIoU"]
):
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Training Curves — DeepLabV3+ on CMP Facade Dataset", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("chart_training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: chart_training_curves.png")
