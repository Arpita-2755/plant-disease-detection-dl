"""Background-bias probe, following Noyan (2022).

A random forest sees only 8 border pixels of each image (4 corners and 4 edge
midpoints). If it beats random guessing, the backgrounds leak label information
and headline accuracy on PlantVillage overstates real-world performance.

  python -m src.bias_probe
"""
from __future__ import annotations

from src import config as C
from src.common import load_split, update_metrics

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def eight_pixels(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] - 1
    m = n // 2
    pts = [(0, 0), (0, m), (0, n), (m, 0), (m, n), (n, 0), (n, m), (n, n)]
    return np.concatenate([x[:, r, c] for r, c in pts], axis=1)


def main() -> None:
    xtr, ytr, _ = load_split("train")
    xte, yte, _ = load_split("test")
    rf = RandomForestClassifier(500, random_state=C.SEED, n_jobs=2, class_weight="balanced")
    rf.fit(eight_pixels(xtr), ytr)
    acc = float(rf.score(eight_pixels(xte), yte))
    chance = float(np.bincount(yte).max() / len(yte))
    print(f"8-pixel probe accuracy {acc:.3f} (majority-class rate {chance:.3f})")
    update_metrics("bias_probe", dict(pixels=8, accuracy=acc, majority_rate=chance,
                                      test_images=len(yte)))


if __name__ == "__main__":
    main()
