"""MobileNetV2 transfer learning: feature extraction, then fine-tuning.

Uses the best head hyperparameters found by src.tune when they are available.

  python -m src.train_transfer
"""
from __future__ import annotations

from src import config as C
from src.common import (callbacks, check_save_load, class_weights, eval_dataset, load_split,
                        plot_history, read_metrics, seed_everything, timed, train_dataset,
                        update_metrics)
from src.models import build_transfer, unfreeze_top

import numpy as np
import tensorflow as tf
from tensorflow import keras


def main() -> None:
    seed_everything()
    T = dict(C.TRANSFER)
    best = read_metrics().get("tuning", {}).get("best")
    batch = C.BATCH
    if best:
        T["head_lr"], T["dropout"], batch = best["lr"], best["dropout"], int(best["batch"])
        print("using tuned hyperparameters:", best, flush=True)
    xtr, ytr, _ = load_split("train")
    xva, yva, _ = load_split("val")
    xte, _, _ = load_split("test")
    va = eval_dataset(xva, yva)
    cw = class_weights(ytr)
    ds, steps = train_dataset(xtr, ytr, batch, aug=True)
    model = build_transfer(T["dropout"])
    total = model.count_params()
    head_params = sum(int(tf.size(w)) for w in model.trainable_weights)
    hist: dict[str, list] = {}
    ck = C.MODELS / "transfer_best.keras"
    with timed("transfer"):
        # Phase 1: the ImageNet backbone is frozen, only the new head learns.
        model.compile(keras.optimizers.Adam(T["head_lr"]), "sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        h1 = model.fit(ds, steps_per_epoch=steps, epochs=1 if C.FAST else T["head_epochs"],
                       verbose=0, validation_data=va, class_weight=cw,
                       callbacks=callbacks("transfer_head", ck, 3, 2)).history
        head_eval = model.evaluate(va, verbose=0)
        # Phase 2: unfreeze the last inverted-residual blocks at a small rate.
        ft_params = unfreeze_top(model, T["unfreeze_from"])
        model.compile(keras.optimizers.Adam(T["ft_lr"]), "sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        h2 = model.fit(ds, steps_per_epoch=steps, epochs=1 if C.FAST else T["ft_epochs"],
                       verbose=0, validation_data=va, class_weight=cw,
                       callbacks=callbacks("transfer_finetune", ck, 3, 2)).history
    for k in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"):
        hist[k] = [float(v) for v in h1.get(k, []) + h2.get(k, [])]
    ft_eval = model.evaluate(va, verbose=0)
    update_metrics("transfer", dict(
        params=total, head_params=head_params, finetune_params=ft_params, batch=batch,
        head_lr=T["head_lr"], dropout=T["dropout"], ft_lr=T["ft_lr"],
        unfreeze_from=T["unfreeze_from"], head_epochs_run=len(h1["loss"]),
        ft_epochs_run=len(h2["loss"]), history=hist, val_acc_after_head=head_eval[1],
        val_loss_after_head=head_eval[0], val_acc_after_ft=ft_eval[1], val_loss_after_ft=ft_eval[0]))
    plot_history(hist, "transfer_curves", "MobileNetV2 transfer learning", split_at=len(h1["loss"]))
    check_save_load(model, C.MODELS / "transfer.keras", xte[:64], "transfer")
    ck.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
