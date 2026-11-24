"""Copy one random held-out test image per class into app/samples/.

The choice is seeded and made without looking at model predictions.

  python -m src.export_samples
"""
from __future__ import annotations

import shutil

import pandas as pd

from src import config as C


def main() -> None:
    out = C.ROOT / "app" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.MANIFEST)
    for cls, label in C.CLASS_MAP.items():
        row = df[(df.split == "test") & (df.class_dir == cls)].sample(1, random_state=C.SEED).iloc[0]
        shutil.copy(C.DATA_DIR / cls / row.filename, out / (label.replace(" - ", "_").replace(" ", "_") + ".jpg"))
    print(len(list(out.glob("*.jpg"))), "samples in", out.relative_to(C.ROOT))


if __name__ == "__main__":
    main()
