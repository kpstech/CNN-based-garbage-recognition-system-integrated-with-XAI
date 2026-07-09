"""
TrashNet — Evaluation Plots
============================
Reads saved history from train_model.py and generates:
  1. Train vs Val Accuracy
  2. Train vs Val Loss
  3. Val Accuracy vs Val Loss (dual axis)
  4. Confusion Matrix
  5. ROC Curve + AUC (one-vs-rest)
  6. Precision-Recall Curve

Run AFTER train_model.py:
    python plot.py

All plots saved to: model/plots/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# LOAD SAVED DATA
# ─────────────────────────────────────────────
HISTORY_PATH = "model/training_history.json"
SUMMARY_PATH = "model/training_summary.json"
PLOTS_DIR    = "model/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading saved training data...")

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(f"Run train_model.py first. '{HISTORY_PATH}' not found.")

with open(HISTORY_PATH) as f:
    hist = json.load(f)

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

CLASSES    = summary["classes"]
N_CLASSES  = len(CLASSES)
p1_end     = hist["phase1_end_epoch"]     # epoch where phase 1 ended
merged     = hist["merged"]               # full merged history
y_true     = np.array(summary["y_true"])  # true test labels
y_pred     = np.array(summary["y_pred"])  # predicted test labels
y_prob     = np.array(summary["y_prob"])  # predicted probabilities (N × 6)
test_acc   = summary["test_accuracy"]

print(f"  Classes        : {CLASSES}")
print(f"  Phase 1 epochs : {hist['phase1']['epochs']}")
print(f"  Phase 2 epochs : {hist['phase2']['epochs']}")
print(f"  Test accuracy  : {test_acc}%")
print(f"  Test samples   : {len(y_true)}")

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "cardboard": "#B45309",
    "glass":     "#2563EB",
    "metal":     "#64748B",
    "paper":     "#D97706",
    "plastic":   "#DC2626",
    "trash":     "#6B7280",
}
TRAIN_COLOR = "#2D6A4F"
VAL_COLOR   = "#52B788"
LOSS_COLOR  = "#E76F51"
LOSS_VAL    = "#F4A261"

plt.rcParams.update({
    "font.family":   "DejaVu Serif",
    "font.size":     11,
    "axes.titlesize":13,
    "axes.labelsize":11,
    "legend.fontsize":9,
    "figure.dpi":    150,
})

epochs_all = range(1, len(merged["accuracy"]) + 1)

# ─────────────────────────────────────────────
# PLOT 1 — Train vs Val Accuracy
# ─────────────────────────────────────────────
print("\n[1/6] Plotting Train vs Val Accuracy...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_all, merged["accuracy"],     color=TRAIN_COLOR, lw=2, label="Train Accuracy")
ax.plot(epochs_all, merged["val_accuracy"], color=VAL_COLOR,   lw=2, ls="--", label="Val Accuracy")
ax.axvline(p1_end, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"Phase 1→2 (epoch {p1_end})")
ax.axhline(0.92, color="red", ls="--", lw=1, alpha=0.5, label="92% baseline")
ax.fill_between(epochs_all, merged["accuracy"], merged["val_accuracy"],
                alpha=0.08, color=TRAIN_COLOR)
ax.set_title("Train vs Validation Accuracy — EfficientNetB0 on TrashNet")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_accuracy_curve.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/01_accuracy_curve.png")

# ─────────────────────────────────────────────
# PLOT 2 — Train vs Val Loss
# ─────────────────────────────────────────────
print("[2/6] Plotting Train vs Val Loss...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_all, merged["loss"],     color=LOSS_COLOR, lw=2, label="Train Loss")
ax.plot(epochs_all, merged["val_loss"], color=LOSS_VAL,   lw=2, ls="--", label="Val Loss")
ax.axvline(p1_end, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"Phase 1→2 (epoch {p1_end})")
ax.fill_between(epochs_all, merged["loss"], merged["val_loss"],
                alpha=0.08, color=LOSS_COLOR)
ax.set_title("Train vs Validation Loss — EfficientNetB0 on TrashNet")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_loss_curve.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/02_loss_curve.png")

# ─────────────────────────────────────────────
# PLOT 3 — Val Accuracy vs Val Loss (dual axis)
# ─────────────────────────────────────────────
print("[3/6] Plotting Val Accuracy vs Val Loss (dual axis)...")

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(epochs_all, merged["val_accuracy"], color=VAL_COLOR, lw=2, label="Val Accuracy")
ax2.plot(epochs_all, merged["val_loss"],     color=LOSS_VAL,  lw=2, ls="--", label="Val Loss")
ax1.axvline(p1_end, color="gray", ls=":", lw=1.2, alpha=0.7)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Val Accuracy", color=VAL_COLOR)
ax2.set_ylabel("Val Loss",     color=LOSS_VAL)
ax1.tick_params(axis="y", labelcolor=VAL_COLOR)
ax2.tick_params(axis="y", labelcolor=LOSS_VAL)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("Validation Accuracy vs Validation Loss")
ax1.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_val_acc_vs_loss.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/03_val_acc_vs_loss.png")

# ─────────────────────────────────────────────
# PLOT 4 — Confusion Matrix (normalised)
# ─────────────────────────────────────────────
print("[4/6] Plotting Confusion Matrix...")

cm = np.array(summary["confusion_matrix"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=10)
ax.set_yticklabels(CLASSES, fontsize=10)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix (Normalised) — Test Accuracy: {test_acc}%")

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        val = cm_norm[i, j]
        txt = f"{val:.2f}"
        color = "white" if val > 0.60 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold" if i == j else "normal")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_confusion_matrix.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/04_confusion_matrix.png")

# ─────────────────────────────────────────────
# PLOT 5 — ROC Curve + AUC (one-vs-rest)
# ─────────────────────────────────────────────
print("[5/6] Plotting ROC Curves with AUC...")

y_bin = label_binarize(y_true, classes=range(N_CLASSES))

fig, ax = plt.subplots(figsize=(8, 6))

# Per-class ROC
roc_aucs = {}
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    roc_aucs[cls] = roc_auc
    ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=1.8,
            label=f"{cls}  (AUC = {roc_auc:.3f})")

# Micro average
fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
micro_auc = auc(fpr_micro, tpr_micro)
ax.plot(fpr_micro, tpr_micro, color="black", lw=2.2, ls="-.",
        label=f"Micro-avg  (AUC = {micro_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — One-vs-Rest per Class")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.25)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_roc_auc.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/05_roc_auc.png")

# ─────────────────────────────────────────────
# PLOT 6 — Precision-Recall Curve
# ─────────────────────────────────────────────
print("[6/6] Plotting Precision-Recall Curves...")

fig, ax = plt.subplots(figsize=(8, 6))

pr_aucs = {}
for i, cls in enumerate(CLASSES):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_bin[:, i], y_prob[:, i])
    pr_aucs[cls] = ap
    ax.plot(rec, prec, color=CLASS_COLORS[cls], lw=1.8,
            label=f"{cls}  (AP = {ap:.3f})")

# Micro average
prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
ap_micro = average_precision_score(y_bin, y_prob, average="micro")
ax.plot(rec_micro, prec_micro, color="black", lw=2.2, ls="-.",
        label=f"Micro-avg  (AP = {ap_micro:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — One-vs-Rest per Class")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(alpha=0.25)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_precision_recall.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/06_precision_recall.png")

# ─────────────────────────────────────────────
# SUMMARY TABLE — print to console
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION SUMMARY")
print("=" * 60)
print(f"  Test Accuracy  : {test_acc}%")
print(f"\n  Per-class AUC (ROC):")
for cls, a in roc_aucs.items():
    print(f"    {cls:12s}: {a:.4f}")
print(f"  Micro-avg AUC  : {micro_auc:.4f}")
print(f"\n  Per-class AP (Precision-Recall):")
for cls, a in pr_aucs.items():
    print(f"    {cls:12s}: {a:.4f}")
print(f"  Micro-avg AP   : {ap_micro:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

print("=" * 60)
print("  ALL PLOTS SAVED")
print("=" * 60)
print(f"  {PLOTS_DIR}/01_accuracy_curve.png")
print(f"  {PLOTS_DIR}/02_loss_curve.png")
print(f"  {PLOTS_DIR}/03_val_acc_vs_loss.png")
print(f"  {PLOTS_DIR}/04_confusion_matrix.png")
print(f"  {PLOTS_DIR}/05_roc_auc.png")
print(f"  {PLOTS_DIR}/06_precision_recall.png")
print("=" * 60)