"""Held-out test evaluation and baseline vs transfer comparison.

  python -m src.evaluate
"""
from __future__ import annotations

import time

from src import config as C
from src.common import (BLUE, ORANGE, image_grid, load_split, predict, savefig, to_float,
                        update_metrics)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from tensorflow import keras

TITLES = {"baseline": "Baseline CNN", "transfer": "MobileNetV2 transfer"}


def main() -> None:
    x, y, df = load_split("test")
    out, preds = {}, {}
    for name in ("baseline", "transfer"):
        model = keras.models.load_model(C.MODELS / f"{name}.keras")
        p = predict(model, x)
        preds[name] = p
        yh = p.argmax(1)
        acc = accuracy_score(y, yh)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="macro", zero_division=0)
        ci = binomtest(int((yh == y).sum()), len(y)).proportion_ci(method="wilson")
        one = tf.constant(to_float(x[:1]))
        graph = tf.function(lambda im: model(im, training=False))     # compiled inference
        for _ in range(3):
            graph(one)
        t = time.perf_counter()
        for _ in range(30):
            graph(one)
        ms = (time.perf_counter() - t) / 30 * 1000
        rep = classification_report(y, yh, labels=range(C.NUM_CLASSES), target_names=C.CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        print(f"\n=== {TITLES[name]}: accuracy {acc:.4f} ===")
        print(classification_report(y, yh, labels=range(C.NUM_CLASSES), target_names=C.CLASS_NAMES,
                                    digits=3, zero_division=0))
        cm = confusion_matrix(y, yh, labels=range(C.NUM_CLASSES))
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, linewidths=0.5,
                    xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, annot_kws={"size": 8})
        ax.set(xlabel="Predicted class", ylabel="True class",
               title=f"{TITLES[name]} confusion matrix ({len(y)} test images)")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", labelsize=8)
        savefig(fig, f"confusion_{name}")
        wrong = np.where(yh != y)[0]
        pairs = {}
        for i in wrong:
            k = f"{C.CLASS_NAMES[y[i]]} -> {C.CLASS_NAMES[yh[i]]}"
            pairs[k] = pairs.get(k, 0) + 1
        out[name] = dict(
            accuracy=acc, macro_precision=pr, macro_recall=rc, macro_f1=f1,
            correct=int((yh == y).sum()), total=len(y), errors=len(wrong),
            wilson95=[ci.low, ci.high], latency_ms=round(ms, 1),
            mean_conf_correct=float(p.max(1)[yh == y].mean()),
            mean_conf_wrong=float(p.max(1)[yh != y].mean()) if len(wrong) else None,
            per_class={c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                       for c in C.CLASS_NAMES},
            top_confusions=dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:8]))
        worst = wrong[np.argsort(-p[wrong].max(1))][:10]
        if len(worst):
            image_grid([x[i] for i in worst],
                       [f"true: {C.CLASS_NAMES[y[i]]}\npred: {C.CLASS_NAMES[yh[i]]} ({p[i].max():.2f})"
                        .replace("Yellow leaf curl virus", "YLCV") for i in worst],
                       5, f"misclassified_{name}", size=2.1, title_size=6.5, title_room=0.5)

    a = preds["baseline"].argmax(1) == y
    b = preds["transfer"].argmax(1) == y
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    out["mcnemar"] = dict(only_baseline_correct=only_a, only_transfer_correct=only_b,
                          both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
                          p_value=float(binomtest(only_a, only_a + only_b).pvalue)
                          if only_a + only_b else 1.0)
    update_metrics("evaluation", out)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    pos = np.arange(4)
    for off, n, col in ((-0.2, "baseline", BLUE), (0.2, "transfer", ORANGE)):
        vals = [out[n][k] for k in keys]
        axes[0].bar(pos + off, vals, 0.37, color=col, label=TITLES[n])
        for p_, v in zip(pos + off, vals):
            axes[0].text(p_, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, color="#52514e")
    axes[0].set(title="Test-set summary", ylim=(0, 1.1))
    axes[0].set_xticks(pos, ["Accuracy", "Precision", "Recall", "F1"])
    pos = np.arange(C.NUM_CLASSES)
    for off, n, col in ((-0.2, "baseline", BLUE), (0.2, "transfer", ORANGE)):
        axes[1].barh(pos + off, [out[n]["per_class"][c]["f1-score"] for c in C.CLASS_NAMES],
                     0.37, color=col, label=TITLES[n])
    axes[1].set_yticks(pos, C.CLASS_NAMES, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(title="F1 score per class", xlim=(0, 1.02))
    for ax, g in ((axes[0], "x"), (axes[1], "y")):
        ax.grid(axis=g, visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    savefig(fig, "model_comparison")


if __name__ == "__main__":
    main()
