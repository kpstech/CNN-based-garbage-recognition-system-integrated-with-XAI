"""
TrashNet Garbage Classification — Training Script (Fixed)
==========================================================
Fixes applied:
  1. ModelCheckpoint — save_weights_only=True  (fixes 'options' error)
  2. decode_image with expand_animations=False  (fixes shape error)
  3. Model.save fallback to .h5 if .keras fails
  4. All numpy conversions made explicit (no tf.size)
  5. Cache uses memory safely
  6. Full error messages with helpful hints

Run:
    python train_model.py

Outputs → model/
    garbage_classifier.keras
    class_names.json
    training_history.json    ← used by plot.py
    training_summary.json    ← used by plot.py
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# CONFIG  (all document-specified values)
# ─────────────────────────────────────────────
CFG = {
    "dataset_path":    "dataset/trashnet",
    "model_dir":       "model",
    "model_path":      "model/garbage_classifier.keras",
    "history_path":    "model/training_history.json",
    "summary_path":    "model/training_summary.json",
    "classnames_path": "model/class_names.json",

    "classes":         ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "img_size":        (224, 224),
    "batch_size":      32,
    "seed":            42,

    # Augmentation — document values
    "aug_rotation":    0.25,
    "aug_zoom":        0.20,
    "aug_contrast":    0.20,
    "aug_translation": 0.10,

    # Phase 1 — head training
    "p1_lr":           1e-3,
    "p1_epochs":       25,
    "p1_patience":     5,
    "p1_lr_factor":    0.3,
    "p1_lr_patience":  3,

    # Phase 2 — fine-tuning
    "p2_lr":           1e-5,
    "p2_epochs":       15,
    "p2_patience":     6,
    "p2_unfreeze":     20,
    "p2_lr_factor":    0.2,
    "p2_lr_patience":  3,

    # Head — document specification
    "dropout":         0.50,
    "dense_units":     256,
}

os.makedirs(CFG["model_dir"], exist_ok=True)
tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])
AUTOTUNE = tf.data.AUTOTUNE

print("=" * 60)
print("  TrashNet — EfficientNetB0  |  Fixed Version")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"  GPU        : {gpus[0].name if gpus else 'None — CPU mode'}")

# ─────────────────────────────────────────────
# 1. LOAD FILE PATHS + LABELS
# ─────────────────────────────────────────────
print("\n[1/8] Scanning dataset folders...")

all_paths, all_labels = [], []

for idx, cls in enumerate(CFG["classes"]):
    folder = os.path.join(CFG["dataset_path"], cls)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"\nFolder not found: {os.path.abspath(folder)}\n"
            f"Expected structure:\n"
            f"  {CFG['dataset_path']}/\n"
            f"    cardboard/\n"
            f"    glass/\n"
            f"    metal/\n"
            f"    paper/\n"
            f"    plastic/\n"
            f"    trash/"
        )
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    all_paths.extend(files)
    all_labels.extend([idx] * len(files))
    print(f"  {cls:12s} : {len(files):4d} images")

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"\n  Total : {len(all_paths)} images across {len(CFG['classes'])} classes")

if len(all_paths) == 0:
    raise ValueError("No images found. Check dataset path and that images are .jpg/.png")

# ─────────────────────────────────────────────
# 2. STRATIFIED 80 / 15 / 5 SPLIT
# ─────────────────────────────────────────────
print("\n[2/8] Stratified 80/15/5 split...")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.20,
    stratify=all_labels,
    random_state=CFG["seed"]
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.25,            # 25% of 20% = 5% of total
    stratify=temp_labels,
    random_state=CFG["seed"]
)

print(f"  Train : {len(train_paths):4d}  ({len(train_paths)/len(all_paths)*100:.1f}%)")
print(f"  Val   : {len(val_paths):4d}  ({len(val_paths)/len(all_paths)*100:.1f}%)")
print(f"  Test  : {len(test_paths):4d}  ({len(test_paths)/len(all_paths)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. CLASS WEIGHTS  (handles Trash imbalance)
# ─────────────────────────────────────────────
print("\n[3/8] Computing class weights...")

cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = {i: float(w) for i, w in enumerate(cw)}
for i, cls in enumerate(CFG["classes"]):
    print(f"  {cls:12s} : {class_weights[i]:.4f}")

# ─────────────────────────────────────────────
# 4. tf.data PIPELINE  (with RAM cache)
# ─────────────────────────────────────────────
print("\n[4/8] Building tf.data pipeline...")

def load_img(path, label):
    raw = tf.io.read_file(path)
    # expand_animations=False ensures shape is always (H, W, 3)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, CFG["img_size"])
    img = tf.cast(img, tf.float32)
    return img, label

aug_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(CFG["aug_rotation"]),
    layers.RandomZoom(CFG["aug_zoom"]),
    layers.RandomContrast(CFG["aug_contrast"]),
    layers.RandomTranslation(CFG["aug_translation"], CFG["aug_translation"]),
], name="augmentation")

def augment_fn(img, label):
    return aug_layer(img, training=True), label

def make_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths.tolist(), [int(l) for l in labels])
    )
    if training:
        ds = ds.shuffle(len(paths), seed=CFG["seed"], reshuffle_each_iteration=True)
    ds = ds.map(load_img, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                    # loads from disk once, then RAM
    if training:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(CFG["batch_size"]).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_paths, train_labels, training=True)
val_ds   = make_ds(val_paths,   val_labels,   training=False)
test_ds  = make_ds(test_paths,  test_labels,  training=False)

# Quick sanity check
print("  Verifying one batch...")
for imgs, lbls in train_ds.take(1):
    print(f"  Image batch shape : {imgs.shape}")
    print(f"  Label batch shape : {lbls.shape}")
    print(f"  Pixel min/max     : {imgs.numpy().min():.1f} / {imgs.numpy().max():.1f}")
    print(f"  Unique labels     : {sorted(np.unique(lbls.numpy()).tolist())}")

# ─────────────────────────────────────────────
# 5. BUILD MODEL  (document architecture)
# ─────────────────────────────────────────────
print("\n[5/8] Building EfficientNetB0 model...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3), name="input")
x      = tf.keras.applications.efficientnet.preprocess_input(inputs)
x      = base_model(x, training=False)                          # 7×7×1280
x      = layers.GlobalAveragePooling2D(name="gap")(x)           # → 1280
x      = layers.BatchNormalization(name="bn")(x)                # stabilise
x      = layers.Dense(CFG["dense_units"], activation="relu",
                       name="dense_256")(x)                     # Dense(256,ReLU)
x      = layers.Dropout(CFG["dropout"], name="dropout")(x)      # Dropout(0.5)
outputs= layers.Dense(len(CFG["classes"]), activation="softmax",
                       name="output")(x)                        # Dense(6,Softmax)

model  = keras.Model(inputs, outputs, name="TrashNet_EfficientNetB0")

total_p     = model.count_params()
trainable_p = int(sum(np.prod(v.shape) for v in model.trainable_weights))
print(f"  Total params       : {total_p:,}")
print(f"  Trainable (Phase 1): {trainable_p:,}")
print(f"  Backbone layers    : {len(base_model.layers)}")

# ─────────────────────────────────────────────
# 6. PHASE 1 — HEAD TRAINING
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  [6/8] PHASE 1 — Head training")
print(f"  Backbone : FROZEN  |  LR = {CFG['p1_lr']}")
print(f"  Max epochs: {CFG['p1_epochs']}  |  Batch: {CFG['batch_size']}")
print("="*55)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CFG["p1_lr"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# FIX: save_weights_only=True avoids the 'options' argument error
callbacks_p1 = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=CFG["p1_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=CFG["p1_lr_factor"],
        patience=CFG["p1_lr_patience"],
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="model/phase1_best.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,      # ← KEY FIX: avoids 'options' error
        verbose=1
    )
]

t0 = time.time()
h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["p1_epochs"],
    class_weight=class_weights,
    callbacks=callbacks_p1
)
p1_time     = time.time() - t0
p1_best_val = float(max(h1.history["val_accuracy"]))
p1_eps      = len(h1.history["accuracy"])

print(f"\n  Phase 1 done  →  {p1_time/60:.1f} min  |  {p1_eps} epochs")
print(f"  Best val accuracy : {p1_best_val*100:.2f}%")

# ─────────────────────────────────────────────
# 7. PHASE 2 — FINE-TUNING
# ─────────────────────────────────────────────
print("\n" + "="*55)
print(f"  [7/8] PHASE 2 — Fine-tuning last {CFG['p2_unfreeze']} layers")
print(f"  LR = {CFG['p2_lr']}  (10× smaller than Phase 1)")
print(f"  Max epochs: {CFG['p2_epochs']}")
print("="*55)

base_model.trainable = True
for layer in base_model.layers[:-CFG["p2_unfreeze"]]:
    layer.trainable = False

frozen   = sum(1 for l in base_model.layers if not l.trainable)
unfrozen = sum(1 for l in base_model.layers if l.trainable)
p2_tp    = int(sum(np.prod(v.shape) for v in model.trainable_weights))
print(f"  Frozen   : {frozen} layers")
print(f"  Unfrozen : {unfrozen} layers")
print(f"  Trainable params : {p2_tp:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CFG["p2_lr"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=CFG["p2_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=CFG["p2_lr_factor"],
        patience=CFG["p2_lr_patience"],
        min_lr=1e-8,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="model/phase2_best.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,      # ← KEY FIX
        verbose=1
    )
]

t0 = time.time()
h2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG["p2_epochs"],
    class_weight=class_weights,
    callbacks=callbacks_p2
)
p2_time     = time.time() - t0
p2_best_val = float(max(h2.history["val_accuracy"]))
p2_eps      = len(h2.history["accuracy"])

print(f"\n  Phase 2 done  →  {p2_time/60:.1f} min  |  {p2_eps} epochs")
print(f"  Best val accuracy : {p2_best_val*100:.2f}%")

# ─────────────────────────────────────────────
# 8a. SAVE MODEL
# ─────────────────────────────────────────────
print("\n[8/8] Saving model and evaluation data...")

# Try .keras first, fallback to .h5
saved_path = CFG["model_path"]
try:
    model.save(CFG["model_path"])
    print(f"  Model saved (.keras) → {CFG['model_path']}")
except Exception as e:
    saved_path = CFG["model_path"].replace(".keras", ".h5")
    model.save(saved_path)
    print(f"  .keras failed ({type(e).__name__}), saved .h5 → {saved_path}")

# Save class names
with open(CFG["classnames_path"], "w") as f:
    json.dump(CFG["classes"], f, indent=2)

# ─────────────────────────────────────────────
# 8b. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

y_true, y_pred, y_prob = [], [], []
for imgs, labels in test_ds:
    probs = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_prob.extend(probs.tolist())
    y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CFG["classes"]))
cm = confusion_matrix(y_true, y_pred)

# ─────────────────────────────────────────────
# 8c. SAVE HISTORY + SUMMARY  (for plot.py)
# ─────────────────────────────────────────────
history_data = {
    "phase1": {
        "accuracy":     [float(x) for x in h1.history["accuracy"]],
        "val_accuracy": [float(x) for x in h1.history["val_accuracy"]],
        "loss":         [float(x) for x in h1.history["loss"]],
        "val_loss":     [float(x) for x in h1.history["val_loss"]],
        "epochs":       p1_eps,
        "time_min":     round(p1_time / 60, 2)
    },
    "phase2": {
        "accuracy":     [float(x) for x in h2.history["accuracy"]],
        "val_accuracy": [float(x) for x in h2.history["val_accuracy"]],
        "loss":         [float(x) for x in h2.history["loss"]],
        "val_loss":     [float(x) for x in h2.history["val_loss"]],
        "epochs":       p2_eps,
        "time_min":     round(p2_time / 60, 2)
    },
    "merged": {
        "accuracy":     [float(x) for x in h1.history["accuracy"] + h2.history["accuracy"]],
        "val_accuracy": [float(x) for x in h1.history["val_accuracy"] + h2.history["val_accuracy"]],
        "loss":         [float(x) for x in h1.history["loss"] + h2.history["loss"]],
        "val_loss":     [float(x) for x in h1.history["val_loss"] + h2.history["val_loss"]],
    },
    "phase1_end_epoch": p1_eps
}

with open(CFG["history_path"], "w") as f:
    json.dump(history_data, f, indent=2)

summary = {
    "model":                "EfficientNetB0",
    "dataset":              "TrashNet",
    "total_images":         int(len(all_paths)),
    "split":                "80/15/5 stratified",
    "train_images":         int(len(train_paths)),
    "val_images":           int(len(val_paths)),
    "test_images":          int(len(test_paths)),
    "classes":              CFG["classes"],
    "batch_size":           CFG["batch_size"],
    "img_size":             list(CFG["img_size"]),
    "phase1_best_val_acc":  round(p1_best_val * 100, 2),
    "phase2_best_val_acc":  round(p2_best_val * 100, 2),
    "test_accuracy":        round(float(test_acc) * 100, 2),
    "test_loss":            round(float(test_loss), 4),
    "total_train_min":      round((p1_time + p2_time) / 60, 2),
    "confusion_matrix":     cm.tolist(),
    "y_true":               y_true.tolist(),
    "y_pred":               y_pred.tolist(),
    "y_prob":               [[round(float(p), 5) for p in row] for row in y_prob]
}

with open(CFG["summary_path"], "w") as f:
    json.dump(summary, f, indent=2)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"  Phase 1 best val acc : {p1_best_val*100:.2f}%")
print(f"  Phase 2 best val acc : {p2_best_val*100:.2f}%")
print(f"  Test accuracy        : {test_acc*100:.2f}%")
print(f"  Training time        : {(p1_time+p2_time)/60:.1f} min")
print(f"\n  Files saved:")
print(f"    {saved_path}")
print(f"    {CFG['history_path']}")
print(f"    {CFG['summary_path']}")
print(f"    {CFG['classnames_path']}")
print(f"\n  Run next:  python plot.py")
print("=" * 60)