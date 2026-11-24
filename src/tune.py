"""CPU-friendly hyperparameter search on cached MobileNetV2 features.

The frozen backbone gives the same features every epoch, so they are computed
once and only the small classification head is trained for each grid point.
A second experiment compares optimizers on the same head (Unit I).

  python -m src.tune
"""
from __future__ import annotations

import itertools

from src import config as C
from src.common import (AQUA, BLUE, ORANGE, VIOLET, YELLOW, class_weights, load_split, savefig,
                        seed_everything, timed, to_float, update_metrics)
from src.models import build_transfer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers as L


def extract(model, x):
    feat = keras.Model(model.input, model.get_layer("gap").output)
    return np.concatenate([feat(to_float(x[i:i + 64]), training=False).numpy()
                           for i in range(0, len(x), 64)])


def train_head(ftr, ytr, fva, yva, optimizer, dropout, batch, epochs):
    seed_everything()
    m = keras.Sequential([keras.Input(ftr.shape[1:]), L.Dropout(dropout),
                          L.Dense(C.NUM_CLASSES, activation="softmax")])
    m.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m.fit(ftr, ytr, batch_size=batch, epochs=epochs, verbose=0,
                 validation_data=(fva, yva), class_weight=class_weights(ytr)).history


def main() -> None:
    seed_everything()
    xtr, ytr, _ = load_split("train")
    xva, yva, _ = load_split("val")
    epochs = 1 if C.FAST else C.TUNE["epochs"]
    with timed("tuning"):
        base = build_transfer(0.0)
        ftr, fva = extract(base, xtr), extract(base, xva)
        rows = []
        for lr, do, bs in itertools.product(C.TUNE["lr"], C.TUNE["dropout"], C.TUNE["batch"]):
            h = train_head(ftr, ytr, fva, yva, keras.optimizers.Adam(lr), do, bs, epochs)
            e = int(np.argmin(h["val_loss"]))
            rows.append(dict(lr=lr, dropout=do, batch=bs, val_acc=float(h["val_accuracy"][e]),
                             val_loss=float(h["val_loss"][e]), best_epoch=e + 1,
                             train_acc=float(h["accuracy"][e])))
            print(rows[-1], flush=True)
        df = pd.DataFrame(rows).sort_values(["val_loss"]).reset_index(drop=True)
        best = df.iloc[0].to_dict()

        opt_hist = {}
        for name, opt in (("SGD", keras.optimizers.SGD(1e-2)),
                          ("SGD + momentum", keras.optimizers.SGD(1e-2, momentum=0.9)),
                          ("RMSprop", keras.optimizers.RMSprop(1e-3)),
                          ("Adagrad", keras.optimizers.Adagrad(1e-2)),
                          ("Adam", keras.optimizers.Adam(1e-3))):
            h = train_head(ftr, ytr, fva, yva, opt, 0.2, 32, epochs)
            opt_hist[name] = {k: [float(v) for v in vs] for k, vs in h.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for (name, h), color in zip(opt_hist.items(), (BLUE, ORANGE, AQUA, YELLOW, VIOLET)):
        ep = np.arange(1, epochs + 1)
        axes[0].plot(ep, h["loss"], color=color, label=name)
        axes[1].plot(ep, h["val_accuracy"], color=color, label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "optimizer_comparison")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    lab = [f"lr={r.lr:g}, drop={r.dropout:g}, batch={int(r.batch)}" for r in df.itertuples()]
    ax.barh(lab, df.val_loss, color=[ORANGE] + [BLUE] * (len(df) - 1), height=0.62)
    for i, r in enumerate(df.itertuples()):
        ax.text(r.val_loss, i, f"  {r.val_loss:.3f}  (acc {r.val_acc:.3f})", va="center",
                fontsize=8, color="#52514e")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Best validation loss (lower is better)", xlim=(0, df.val_loss.max() * 1.3),
           title="Grid search, best configuration first")
    savefig(fig, "tuning_results")
    update_metrics("tuning", dict(table=df.to_dict("records"), best=best, optimizers={
        k: dict(final_train_loss=v["loss"][-1], final_val_acc=v["val_accuracy"][-1],
                best_val_acc=max(v["val_accuracy"])) for k, v in opt_hist.items()}))


if __name__ == "__main__":
    main()
