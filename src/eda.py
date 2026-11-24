"""Exploratory data analysis plus the Unit I tensor demonstration.

  python -m src.eda
"""
from __future__ import annotations

from src import config as C
from src.common import *  # noqa: F401,F403
from src.common import augment, image_grid, load_manifest, load_split, savefig, update_metrics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def tensor_demo(x: np.ndarray, y: np.ndarray) -> dict:
    """Shapes, ranks, dtypes, basic ops and automatic differentiation."""
    img = tf.convert_to_tensor(x[0])
    batch = tf.cast(tf.convert_to_tensor(x[:32]), tf.float32) / 255.0
    label = tf.constant(int(y[0]))
    onehot = tf.one_hot(y[:32], C.NUM_CLASSES)
    rows = []
    for name, t in (("label", label), ("mean RGB", tf.reduce_mean(batch, [0, 1, 2])),
                    ("one-hot labels", onehot), ("single image", img), ("image batch", batch)):
        rows.append(dict(tensor=name, rank=int(tf.rank(t)), shape=str(tuple(t.shape)),
                         dtype=t.dtype.name, elements=int(tf.size(t))))
    v = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        f = v**2 + 2 * v
    g = float(tape.gradient(f, v))          # d/dv (v^2 + 2v) = 2v + 2 = 8
    flat = tf.reshape(batch[:2], [2, -1])
    gram = tf.matmul(flat, flat, transpose_b=True)
    print(pd.DataFrame(rows).to_string(index=False))
    return dict(tensors=rows, gradient_at_3=g, matmul_shape=list(gram.shape),
                mean_rgb=[round(float(m), 4) for m in tf.reduce_mean(batch, [0, 1, 2])],
                tf_version=tf.__version__)


def main() -> None:
    df = load_manifest()
    counts = pd.crosstab(df.class_dir, df.split).loc[C.CLASS_DIRS, ["train", "val", "test"]]
    counts.index = C.CLASS_NAMES

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    left = np.zeros(len(counts))
    for col, color, lab in (("train", "#2a78d6", "Train"), ("val", "#eb6834", "Validation"),
                            ("test", "#1baf7a", "Test")):
        ax.barh(counts.index, counts[col], left=left, color=color, label=lab, height=0.62,
                edgecolor="white", linewidth=1.5)
        left += counts[col].to_numpy()
    for i, t in enumerate(left):
        ax.text(t + 4, i, int(t), va="center", fontsize=9, color="#52514e")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Images", xlim=(0, max(left) * 1.12), title="Images per class and split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    savefig(fig, "eda_class_distribution")

    # sample grid: 3 training images from each class
    xtr, ytr, dtr = load_split("train")
    rng = np.random.default_rng(C.SEED)
    imgs, titles = [], []
    for r in range(2):
        for c in range(C.NUM_CLASSES):
            idx = rng.choice(np.where(ytr == c)[0])
            imgs.append(xtr[idx])
            titles.append(C.CLASS_NAMES[c].replace(" - ", "\n").replace("Yellow leaf curl virus", "Yellow leaf curl"))
    order = [r * C.NUM_CLASSES + c for half in (0, 5) for r in range(2) for c in range(half, half + 5)]
    image_grid([imgs[i] for i in order], [titles[i] for i in order], 5, "eda_sample_grid")

    # raw image statistics
    stats = []
    for r in df.itertuples():
        p = C.DATA_DIR / r.class_dir / r.filename
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize((64, 64)), np.float32)
            stats.append((im.size[0], im.size[1], p.stat().st_size / 1024, a.mean(), *a.mean((0, 1))))
    st = pd.DataFrame(stats, columns=["w", "h", "kb", "bright", "r", "g", "b"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    axes[0].hist(st.kb, bins=30, color="#2a78d6", edgecolor="white")
    axes[0].set(title="File size", xlabel="KB", ylabel="Images")
    axes[1].hist(st.bright, bins=30, color="#2a78d6", edgecolor="white")
    axes[1].set(title="Mean brightness", xlabel="Mean pixel value (0-255)")
    for col, color, lab in (("r", "#e34948", "Red"), ("g", "#008300", "Green"), ("b", "#2a78d6", "Blue")):
        axes[2].hist(st[col], bins=40, histtype="step", color=color, lw=2, label=lab)
    axes[2].set(title="Mean channel value", xlabel="0-255")
    axes[2].legend()
    fig.tight_layout()
    savefig(fig, "eda_image_stats")

    # augmentation preview
    src = np.repeat(xtr[:1], 1, 0)
    prev, names = [xtr[0]], ["original"]
    for k in range(9):
        a, _ = augment(tf.constant(1000 + k, tf.int64), tf.constant(src), tf.constant([0]))
        prev.append(a[0].numpy())
        names.append(f"augmented {k + 1}")
    image_grid(prev, names, 5, "eda_augmentation")

    leaves = df.groupby("class_dir").leaf_id.nunique().loc[C.CLASS_DIRS]
    leak = int(df.groupby(["class_dir", "leaf_id"]).split.nunique().max())
    demo = tensor_demo(xtr, ytr)
    update_metrics("data", dict(
        total=len(df), per_split=df.split.value_counts().to_dict(),
        counts={n: {k: int(v) for k, v in row.items()} for n, row in counts.iterrows()},
        leaf_groups={n: int(v) for n, v in zip(C.CLASS_NAMES, leaves)},
        max_splits_per_leaf=leak,
        duplicate_files=int(df.sha256_16.duplicated().sum()),
        ungrouped_images=int(df.leaf_id.str.startswith("single:").sum()),
        image_size=[f"{w}x{h}" for w, h in st[["w", "h"]].drop_duplicates().to_numpy()],
        mean_file_kb=round(float(st.kb.mean()), 2), mean_brightness=round(float(st.bright.mean()), 2),
        mean_rgb_255=[round(float(st[c].mean()), 2) for c in "rgb"], tensor_demo=demo))


if __name__ == "__main__":
    main()
