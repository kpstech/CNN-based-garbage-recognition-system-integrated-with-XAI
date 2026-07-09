"""
Misclassified Images Viewer
Run after train_model.py:  python misclassified.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── load saved data ───────────────────────────────────────
with open("model/training_summary.json") as f:
    summary = json.load(f)

CLASSES = summary["classes"]
y_true  = np.array(summary["y_true"])
y_pred  = np.array(summary["y_pred"])

# We need the test image paths — load from summary or re-split
# Re-split with same seed to get same test paths
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/trashnet"
SEED = 42

all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, cls)
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(folder, fname))
            all_labels.append(idx)

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)

# Same split as training
_, temp_paths, _, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.20,
    stratify=all_labels, random_state=SEED
)
_, test_paths, _, _ = train_test_split(
    temp_paths, temp_labels, test_size=0.25,
    stratify=temp_labels, random_state=SEED
)

# ── find misclassified ────────────────────────────────────
wrong_idx = np.where(y_true != y_pred)[0]
print(f"Total test images    : {len(y_true)}")
print(f"Misclassified        : {len(wrong_idx)}")
print(f"Correctly classified : {len(y_true) - len(wrong_idx)}")
print(f"Test accuracy        : {(1 - len(wrong_idx)/len(y_true))*100:.1f}%")

if len(wrong_idx) == 0:
    print("No misclassified images found!")
    exit()

# ── plot grid ────────────────────────────────────────────
COLS     = 4
MAX_SHOW = min(len(wrong_idx), 20)   # show up to 20
ROWS     = (MAX_SHOW + COLS - 1) // COLS

fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.5, ROWS * 3.8))
fig.suptitle(
    f"Misclassified Images  ({len(wrong_idx)} of {len(y_true)} test images)",
    fontsize=14, fontweight="bold", y=1.01
)
axes = axes.flatten()

for plot_i, wrong_i in enumerate(wrong_idx[:MAX_SHOW]):
    img_path   = test_paths[wrong_i]
    true_label = CLASSES[y_true[wrong_i]]
    pred_label = CLASSES[y_pred[wrong_i]]

    img = Image.open(img_path).convert("RGB")

    ax = axes[plot_i]
    ax.imshow(img)
    ax.set_title(
        f"True  : {true_label}\nPred  : {pred_label}",
        fontsize=9,
        color="red",
        fontweight="bold"
    )
    ax.axis("off")

    # Red border around each wrong image
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(3)
        spine.set_visible(True)

# Hide unused axes
for i in range(MAX_SHOW, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("model/misclassified.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → model/misclassified.png")
