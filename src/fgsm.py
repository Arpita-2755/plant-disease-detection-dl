"""FGSM attack from scratch with tf.GradientTape, plus adversarial training.

  x_adv = clip(x + eps * sign(d loss / d x), 0, 1)

  python -m src.fgsm
"""
from __future__ import annotations

import math

from src import config as C
from src.common import (AQUA, BLUE, ORANGE, INK2, image_grid, load_split, predict, savefig,
                        seed_everything, timed, to_float, update_metrics)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

_xent = keras.losses.SparseCategoricalCrossentropy()


def fgsm_sign(model, x, y):
    """Sign of the input gradient of the loss. x: float [0,1], y: int labels."""
    x = tf.convert_to_tensor(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = _xent(y, model(x, training=False))
    return tf.sign(tape.gradient(loss, x))


def fgsm_attack(model, x, y, eps: float):
    return tf.clip_by_value(x + eps * fgsm_sign(model, x, y), 0.0, 1.0)


def robustness(model, x_uint8, y, bs=32):
    """Accuracy under white-box FGSM for every epsilon (one gradient per image)."""
    hits = np.zeros(len(C.FGSM_EPS))
    for i in range(0, len(y), bs):
        xb, yb = to_float(x_uint8[i:i + bs]), y[i:i + bs]
        sign = fgsm_sign(model, xb, yb).numpy()
        for j, e in enumerate(C.FGSM_EPS):
            adv = np.clip(xb + e * sign, 0, 1)
            hits[j] += (model(adv, training=False).numpy().argmax(1) == yb).sum()
    return [float(h / len(y)) for h in hits]


def adversarial_finetune(model, x_uint8, y, epochs=2, lr=5e-4, eps_max=0.02, bs=32):
    """Half of every batch is replaced by FGSM examples of random strength."""
    opt = keras.optimizers.Adam(lr)

    @tf.function(reduce_retracing=True)
    def step(xb, yb, eps):
        h = tf.shape(xb)[0] // 2
        xin = tf.convert_to_tensor(xb[:h])
        with tf.GradientTape() as t0:
            t0.watch(xin)
            l0 = _xent(yb[:h], model(xin, training=False))
        adv = tf.clip_by_value(xin + eps[:h] * tf.sign(t0.gradient(l0, xin)), 0.0, 1.0)
        xm = tf.concat([adv, xb[h:]], 0)
        with tf.GradientTape() as t1:
            loss = _xent(yb, model(xm, training=True))
        g = t1.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    rng = np.random.default_rng(C.SEED)
    for ep in range(1 if C.FAST else epochs):
        order, losses = rng.permutation(len(y)), []
        for i in range(0, len(y) - bs + 1, bs):
            b = order[i:i + bs]
            xb = to_float(x_uint8[b])
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            eps = rng.uniform(0, eps_max, (len(b), 1, 1, 1)).astype(np.float32)
            losses.append(float(step(tf.constant(xb.copy()), tf.constant(y[b]), tf.constant(eps))))
        print(f"adversarial epoch {ep + 1}: loss {np.mean(losses):.4f}", flush=True)


def main() -> None:
    seed_everything()
    x, y, _ = load_split("test")
    base = keras.models.load_model(C.MODELS / "baseline.keras")
    tran = keras.models.load_model(C.MODELS / "transfer.keras")
    with timed("fgsm", "attack_seconds"):
        acc = {"baseline": robustness(base, x, y), "transfer": robustness(tran, x, y)}
    print(acc, flush=True)

    # --- mitigation: adversarially fine-tune a copy of the baseline
    robust = keras.models.load_model(C.MODELS / "baseline.keras")
    xtr, ytr, _ = load_split("train")
    with timed("fgsm", "adv_training_seconds"):
        adversarial_finetune(robust, xtr, ytr)
    robust.save(C.MODELS / "baseline_robust.keras")
    acc["baseline_adv_trained"] = robustness(robust, x, y)
    update_metrics("fgsm", dict(eps=list(C.FGSM_EPS), accuracy=acc, test_images=len(y)))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k, lab, col, mk in (("baseline", "Baseline CNN", BLUE, "o"),
                            ("transfer", "MobileNetV2 transfer", ORANGE, "s"),
                            ("baseline_adv_trained", "Baseline + adversarial training", AQUA, "^")):
        ax.plot(C.FGSM_EPS, acc[k], color=col, marker=mk, ms=7, label=lab)
    ax.axhline(1 / C.NUM_CLASSES, color=INK2, ls=":", lw=1.2)
    ax.text(C.FGSM_EPS[-1], 1 / C.NUM_CLASSES + 0.012, "random guess", ha="right", fontsize=8, color=INK2)
    ax.set(xlabel="FGSM epsilon (fraction of pixel range)", ylabel="Test accuracy", ylim=(0, 1.02),
           title="Accuracy under white-box FGSM")
    ax.legend(loc="upper right")
    savefig(fig, "fgsm_robustness")

    # --- clean vs adversarial examples for the transfer model
    p = predict(tran, x)
    picks = [int(np.where((y == c) & (p.argmax(1) == c))[0][0]) for c in (2, 7, 9)
             if ((y == c) & (p.argmax(1) == c)).any()]
    ims, tt, colors = [], [], []
    for i in picks:
        im = to_float(x[i:i + 1])
        sign = fgsm_sign(tran, im, y[i:i + 1]).numpy()
        ims += [im[0], (sign[0] + 1) / 2]
        tt += [f"clean\n{C.CLASS_NAMES[y[i]]}\n{p[i].max():.2f}", "sign of gradient\n\n"]
        colors += ["#0b0b0b", "#0b0b0b"]
        for e in (0.005, 0.02, 0.08):
            adv = np.clip(im + e * sign, 0, 1)
            q = tran(adv, training=False).numpy()[0]
            ims.append(adv[0])
            tt.append(f"eps={e}\n{C.CLASS_NAMES[q.argmax()]}\n{q.max():.2f}".replace("Yellow leaf curl virus", "YLCV"))
            colors.append("#0b0b0b" if q.argmax() == y[i] else "#d03b3b")
    if ims:
        image_grid(ims, tt, 5, "fgsm_examples", size=2.0, title_size=7.5, colors=colors)


if __name__ == "__main__":
    main()
