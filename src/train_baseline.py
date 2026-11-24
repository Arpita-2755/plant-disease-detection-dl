"""Train the custom CNN and run the underfit / overfit / regularised experiment.

  python -m src.train_baseline
"""
from __future__ import annotations

from src import config as C
from src.common import (AQUA, BLUE, ORANGE, callbacks, check_save_load, class_weights,
                        eval_dataset, load_split, plot_history, savefig, seed_everything,
                        timed, to_float, train_dataset, update_metrics)
from src.models import build_baseline

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as L


def fit_demo_model(kind: str, xtr, ytr, xva, yva) -> dict:
    """Three 64x64 models on a deliberately small training set."""
    seed_everything()
    inp = keras.Input((64, 64, 3))
    if kind == "underfit":
        x = L.MaxPooling2D(4)(L.Conv2D(4, 3, activation="relu")(inp))
        x = L.GlobalAveragePooling2D()(x)
        epochs, aug = 20, False
    elif kind == "overfit":
        x = L.MaxPooling2D(2)(L.Conv2D(32, 3, activation="relu", padding="same")(inp))
        x = L.MaxPooling2D(2)(L.Conv2D(64, 3, activation="relu", padding="same")(x))
        x = L.Dense(256, activation="relu")(L.Flatten()(x))
        epochs, aug = 20, False
    else:
        x = inp
        for f in (32, 64):
            x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = L.MaxPooling2D(2)(L.ReLU()(L.BatchNormalization(momentum=0.9)(x)))
        x = L.Dropout(0.4)(L.GlobalAveragePooling2D()(x))
        x = L.Dropout(0.3)(L.Dense(64, activation="relu")(x))
        epochs, aug = 30, True
    model = keras.Model(inp, L.Dense(C.NUM_CLASSES, activation="softmax")(x))
    model.compile(keras.optimizers.Adam(1e-3 if kind != "regularised" else 2e-3),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    ds, steps = train_dataset(xtr, ytr, 32, aug=aug)
    cbs = ([keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)]
           if kind == "regularised" else [])
    h = model.fit(ds, steps_per_epoch=steps, epochs=1 if C.FAST else epochs, verbose=0,
                  validation_data=eval_dataset(xva, yva), callbacks=cbs).history
    _, train_acc = model.evaluate(eval_dataset(xtr, ytr), verbose=0)
    va_loss, va_acc = model.evaluate(eval_dataset(xva, yva), verbose=0)
    res = dict(params=model.count_params(), epochs_run=len(h["loss"]), train_acc=train_acc,
               val_acc=va_acc, val_loss=va_loss, gap=train_acc - va_acc, history=h)
    print(kind, {k: round(v, 3) for k, v in res.items() if k != "history"}, flush=True)
    return res


def fit_experiment() -> None:
    xtr, ytr, _ = load_split("train", 64)
    xva, yva, _ = load_split("val", 64)
    rng = np.random.default_rng(C.SEED)
    pick = np.concatenate([rng.choice(np.where(ytr == c)[0], min(40, (ytr == c).sum()), replace=False)
                           for c in range(C.NUM_CLASSES)])
    kinds = ("underfit", "overfit", "regularised")
    res = {k: fit_demo_model(k, xtr[pick], ytr[pick], xva, yva) for k in kinds}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey="row")
    for j, k in enumerate(kinds):
        h = res[k]["history"]
        for i, m in enumerate(("accuracy", "loss")):
            ep = np.arange(1, len(h[m]) + 1)
            axes[i, j].plot(ep, h[m], color=BLUE, label="Training")
            axes[i, j].plot(ep, h["val_" + m], color=ORANGE, label="Validation")
            axes[i, j].set(xlabel="Epoch")
        axes[0, j].set_title(k.capitalize())
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_ylim(0, 3.5)
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "overfit_underfit")
    update_metrics("fit_experiment", {"train_images": int(len(pick)), "results": res})


def main() -> None:
    with timed("fit_experiment"):
        fit_experiment()

    seed_everything()
    xtr, ytr, _ = load_split("train")
    xva, yva, _ = load_split("val")
    xte, _, _ = load_split("test")
    model = build_baseline()
    model.summary()
    model.compile(keras.optimizers.Adam(C.BASELINE["lr"]), "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    ds, steps = train_dataset(xtr, ytr, C.BATCH, aug=True)
    epochs = 1 if C.FAST else C.BASELINE["epochs"]
    with timed("baseline"):
        h = model.fit(ds, steps_per_epoch=steps, epochs=epochs, verbose=0,
                      validation_data=eval_dataset(xva, yva), class_weight=class_weights(ytr),
                      callbacks=callbacks("baseline", C.MODELS / "baseline_best.keras", 8,
                                           cosine_epochs=epochs))
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    update_metrics("baseline", dict(
        params=model.count_params(), epochs_run=len(hist["loss"]), history=hist,
        best_epoch=int(np.argmin(hist["val_loss"]) + 1), best_val_acc=float(
            hist["val_accuracy"][int(np.argmin(hist["val_loss"]))])))
    plot_history(hist, "baseline_curves", "Baseline CNN training")
    check_save_load(model, C.MODELS / "baseline.keras", xte[:64], "baseline")
    (C.MODELS / "baseline_best.keras").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
