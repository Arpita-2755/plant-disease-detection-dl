"""Explainability: CNN activation maps and Grad-CAM heatmaps.

  python -m src.gradcam
"""
from __future__ import annotations

from src import config as C
from src.common import image_grid, load_split, predict, savefig, to_float, update_metrics

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

LAST_CONV = {"baseline": "relu4", "transfer": "out_relu"}


def make_gradcam(model: keras.Model, layer: str):
    """Return f(image[H,W,3] in [0,1], class) -> heatmap[H,W] in [0,1]."""
    grad_model = keras.Model(model.input, [model.get_layer(layer).output, model.output])

    def run(image: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        with tf.GradientTape() as tape:
            maps, probs = grad_model(image[None].astype("float32"), training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(probs[0]))
            # log-probability avoids vanishing gradients when softmax saturates
            score = tf.math.log(probs[:, class_idx] + 1e-8)
        grads = tape.gradient(score, maps)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(maps[0] * weights, axis=-1)).numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return np.clip(cv2.resize(cam, image.shape[1::-1], interpolation=cv2.INTER_CUBIC), 0, 1)

    return run


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def activation_maps(model, image, layers=("relu1", "relu2", "relu3", "relu4"), n=8):
    ext = keras.Model(model.input, [model.get_layer(l).output for l in layers])
    return dict(zip(layers, [m[0].numpy() for m in ext(image[None].astype("float32"))]))


def main() -> None:
    x, y, _ = load_split("test")
    base = keras.models.load_model(C.MODELS / "baseline.keras")
    tran = keras.models.load_model(C.MODELS / "transfer.keras")

    # --- feature maps of the baseline CNN
    idx = int(np.where(y == 2)[0][0])          # a potato early-blight leaf
    img = to_float(x[idx])
    maps = activation_maps(base, img)
    fig, axes = plt.subplots(4, 9, figsize=(11, 5.4))
    for r, (layer, m) in enumerate(maps.items()):
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{layer}\n{m.shape[0]}x{m.shape[1]}x{m.shape[2]}", fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=26)
        strongest = np.argsort(-m.mean((0, 1)))[:8]
        for c, ch in enumerate(strongest, 1):
            axes[r, c].imshow(m[..., ch], cmap="viridis")
    for ax in axes.ravel():
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    fig.suptitle("Baseline CNN activation maps (8 strongest channels per block)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "activation_maps")

    # --- learned first-layer kernels
    k = base.get_layer("conv1").get_weights()[0]
    k = (k - k.min()) / (k.max() - k.min())
    image_grid([cv2.resize(k[..., i], (48, 48), interpolation=cv2.INTER_NEAREST) for i in range(16)],
               [f"kernel {i + 1}" for i in range(16)], 8, "conv1_kernels", size=1.15, title_size=7)

    # --- Grad-CAM: transfer model on correct and incorrect predictions
    p = predict(tran, x)
    yh = p.argmax(1)
    cam_t = make_gradcam(tran, LAST_CONV["transfer"])
    cam_b = make_gradcam(base, LAST_CONV["baseline"])
    right = [int(np.where((y == c) & (yh == c))[0][0]) for c in (0, 2, 3, 5, 7, 8)
             if ((y == c) & (yh == c)).any()]
    wrong_all = np.where(yh != y)[0]
    wrong = [int(i) for i in wrong_all[np.argsort(-p[wrong_all].max(1))][:6]]

    def cam_panel(idxs, name):
        ims, tt = [], []
        for i in idxs:
            im = to_float(x[i])
            ims += [im, overlay(im, cam_t(im))]
            tt += [f"true: {C.CLASS_NAMES[y[i]]}".replace("Yellow leaf curl virus", "YLCV"),
                   f"pred: {C.CLASS_NAMES[yh[i]]} {p[i].max():.2f}".replace("Yellow leaf curl virus", "YLCV")]
        if ims:
            image_grid(ims, tt, 4, name, size=2.3, title_size=7.5)

    cam_panel(right, "gradcam_correct")
    cam_panel(wrong, "gradcam_incorrect")

    # --- side by side: baseline vs transfer attention
    ims, tt = [], []
    for i in right[:4]:
        im = to_float(x[i])
        ims += [im, overlay(im, cam_b(im)), overlay(im, cam_t(im))]
        tt += [C.CLASS_NAMES[y[i]], "baseline Grad-CAM", "transfer Grad-CAM"]
    image_grid(ims, tt, 3, "gradcam_compare", size=2.1, title_size=8)
    update_metrics("gradcam", dict(layers=LAST_CONV, incorrect_shown=len(wrong),
                                   activation_shapes={k: list(v.shape) for k, v in maps.items()}))


if __name__ == "__main__":
    main()
