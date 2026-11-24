"""Shared helpers: seeding, data loading, augmentation, metrics file, plotting."""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager

from src import config as C

import cv2
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------- plotting
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
    "legend.frameon": False,
})


def savefig(fig, name: str) -> None:
    path = C.PLOTS / f"{name}.png"
    fig.savefig(path, facecolor="white")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    print("saved", path.name, flush=True)


def image_grid(images, titles, ncols, name, size=1.7, title_size=8, colors=None):
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.35)))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(images):
            ax.imshow(np.clip(images[i], 0, 1) if images[i].dtype != np.uint8 else images[i])
            ax.set_title(titles[i], fontsize=title_size, fontweight="normal",
                         color=(colors[i] if colors else INK))
    fig.tight_layout()
    savefig(fig, name)


def plot_history(hist: dict, name: str, title: str, split_at: int | None = None):
    """Accuracy and loss curves (two panels, never a dual axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, key, lab in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ep = np.arange(1, len(hist[key]) + 1)
        ax.plot(ep, hist[key], color=BLUE, label="Training")
        ax.plot(ep, hist["val_" + key], color=ORANGE, label="Validation")
        if split_at:
            ax.axvline(split_at + 0.5, color=INK2, ls="--", lw=1)
            ax.text(split_at + 0.6, ax.get_ylim()[0], " fine-tuning", color=INK2,
                    fontsize=8, va="bottom")
        ax.set(xlabel="Epoch", ylabel=lab)
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend(loc="lower right")
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    savefig(fig, name)


# ---------------------------------------------------------------- bookkeeping
def seed_everything(seed: int = C.SEED) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def update_metrics(section: str, values: dict) -> None:
    data = json.loads(C.METRICS.read_text()) if C.METRICS.exists() else {}
    data.setdefault(section, {}).update(values)
    C.METRICS.write_text(json.dumps(data, indent=2, default=float))


def read_metrics() -> dict:
    return json.loads(C.METRICS.read_text()) if C.METRICS.exists() else {}


@contextmanager
def timed(section: str, key: str = "train_seconds"):
    t = time.time()
    yield
    secs = round(time.time() - t, 1)
    update_metrics(section, {key: secs})
    print(f"[{section}] {key} = {secs}s", flush=True)


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        body = " ".join(f"{k}={v:.4f}" for k, v in (logs or {}).items())
        print(f"epoch {epoch + 1} ({time.time() - self.t:.0f}s) {body}", flush=True)


# ---------------------------------------------------------------- data
def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(C.MANIFEST, dtype={"leaf_id": str})
    df["label"] = df.class_dir.map({c: i for i, c in enumerate(C.CLASS_DIRS)})
    if C.FAST:
        df = df.groupby(["class_dir", "split"], group_keys=False).head(6)
    return df


def read_rgb(path, size: int) -> np.ndarray:
    img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_split(split: str, size: int = C.IMG):
    """Return (images uint8 [N,size,size,3], labels int32, manifest rows)."""
    df = load_manifest()
    df = df[df.split == split].reset_index(drop=True)
    x = np.stack([read_rgb(C.DATA_DIR / r.class_dir / r.filename, size)
                  for r in df.itertuples()])
    return x, df.label.to_numpy(np.int32), df


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def augment(step, x, y):
    """Random flip, rotation/zoom/shift, brightness and contrast (training only).

    Stateless ops seeded by the global step make every run reproducible.
    """
    s = tf.stack([tf.constant(C.SEED, tf.int64), step])
    x = tf.cast(x, tf.float32) / 255.0
    n = tf.shape(x)[0]
    size = tf.cast(tf.shape(x)[1], tf.float32)

    def u(k, lo, hi):
        return tf.random.stateless_uniform([n], s + [k, 0], lo, hi)

    flip_h = tf.where(u(1, 0.0, 1.0) < 0.5, -1.0, 1.0)
    flip_v = tf.where(u(2, 0.0, 1.0) < 0.5, -1.0, 1.0)
    ang, zoom = u(3, -0.35, 0.35), u(4, 0.85, 1.10)
    tx, ty = u(5, -0.08, 0.08) * size, u(6, -0.08, 0.08) * size
    c = (size - 1) / 2
    a0, a1 = tf.cos(ang) * zoom * flip_h, -tf.sin(ang) * zoom * flip_v
    b0, b1 = tf.sin(ang) * zoom * flip_h, tf.cos(ang) * zoom * flip_v
    z = tf.zeros([n])
    mat = tf.stack([a0, a1, c - a0 * c - a1 * c + tx,
                    b0, b1, c - b0 * c - b1 * c + ty, z, z], axis=1)
    x = tf.raw_ops.ImageProjectiveTransformV3(
        images=x, transforms=mat, output_shape=tf.shape(x)[1:3], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="REFLECT")
    bright = tf.reshape(u(7, -0.10, 0.10), [-1, 1, 1, 1])
    contrast = tf.reshape(u(8, 0.85, 1.15), [-1, 1, 1, 1])
    mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    return tf.clip_by_value((x - mean) * contrast + mean + bright, 0.0, 1.0), y


def train_dataset(x, y, batch=C.BATCH, aug=True):
    """Infinite, shuffled (and optionally augmented) training pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x), seed=C.SEED).batch(batch).repeat()
    if aug:
        ds = ds.enumerate().map(lambda i, xy: augment(i, *xy), num_parallel_calls=2)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=2)
    return ds.prefetch(2), math.ceil(len(x) / batch)


def eval_dataset(x, y, batch=64):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch)
    return ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b)).prefetch(1)


def predict(model, x_uint8, batch=64) -> np.ndarray:
    return np.concatenate([model(to_float(x_uint8[i:i + batch]), training=False).numpy()
                           for i in range(0, len(x_uint8), batch)])


def class_weights(y) -> dict:
    n = np.bincount(y, minlength=C.NUM_CLASSES)
    return {i: float(len(y) / (C.NUM_CLASSES * max(k, 1))) for i, k in enumerate(n)}


def callbacks(name: str, ckpt, patience: int, reduce_patience: int | None = None,
              cosine_epochs: int | None = None):
    """EarlyStopping + LR schedule + ModelCheckpoint + TensorBoard.

    The schedule is cosine decay when `cosine_epochs` is given, otherwise the
    learning rate is halved whenever the validation loss stops improving.
    """
    if cosine_epochs:
        sched = keras.callbacks.LearningRateScheduler(
            lambda ep, lr: C.BASELINE["lr"] * (0.02 + 0.49 * (1 + math.cos(math.pi * ep / cosine_epochs))))
    else:
        sched = keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=reduce_patience,
                                                  min_lr=1e-6)
    return [
        keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        sched,
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        keras.callbacks.TensorBoard(str(C.LOGS / name), write_graph=True, profile_batch=0),
        EpochLogger(),
    ]


def check_save_load(model, path, x_uint8, section: str) -> None:
    """Save, reload and prove that the reloaded model predicts the same values."""
    before = predict(model, x_uint8)
    model.save(path)
    after = predict(keras.models.load_model(path), x_uint8)
    update_metrics(section, {"save_load": {
        "images_checked": len(x_uint8),
        "max_abs_difference": float(np.abs(before - after).max()),
        "identical_labels": bool((before.argmax(1) == after.argmax(1)).all()),
        "file_mb": round(path.stat().st_size / 1e6, 2)}})
