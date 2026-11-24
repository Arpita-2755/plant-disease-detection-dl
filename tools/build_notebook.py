"""Build (and optionally execute) notebooks/plant_disease_workflow.ipynb.

  python tools/build_notebook.py            write and execute
  python tools/build_notebook.py --no-run   write only
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "plant_disease_workflow.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Plant Leaf Disease Detection - full workflow
**INT422 Deep Learning** - Arpita Mishra (12303670, K2M2359, Roll No. 03)

This notebook walks through the whole project. With `RETRAIN = False` it loads the trained models and
recomputes every evaluation live. Set `RETRAIN = True` to run all training scripts first
(under an hour on a dual-core CPU).
""")
code("""
RETRAIN = False

import os, sys, json
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
from src import config as C
from src.common import *
from src.common import load_split, predict, read_metrics, to_float, train_dataset, eval_dataset
print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| devices:", tf.config.list_physical_devices())
if RETRAIN:
    import subprocess
    subprocess.run([sys.executable, "-m", "src.run_all"], check=True)
metrics = read_metrics()
""")
md("## 1. Data (Units I and III)\n`python -m src.prepare_data` downloads the subset. Images of one leaf never cross splits.")
code("""
df = load_manifest()
display(pd.crosstab(df.class_dir, df.split, margins=True))
print("max splits per leaf:", df.groupby(["class_dir", "leaf_id"]).split.nunique().max())
xtr, ytr, _ = load_split("train"); xva, yva, _ = load_split("val"); xte, yte, _ = load_split("test")
""")
md("### Tensors: rank, shape, dtype, operations and gradients (Unit I)")
code("""
batch = tf.cast(tf.constant(xtr[:32]), tf.float32) / 255.0
for name, t in {"scalar label": tf.constant(int(ytr[0])), "mean RGB": tf.reduce_mean(batch, [0, 1, 2]),
                "one-hot": tf.one_hot(ytr[:32], C.NUM_CLASSES), "image": tf.constant(xtr[0]), "batch": batch}.items():
    print(f"{name:13s} rank={tf.rank(t).numpy()} shape={t.shape} dtype={t.dtype.name}")
v = tf.Variable(3.0)
with tf.GradientTape() as tape:
    f = v ** 2 + 2 * v
print("d/dv (v^2 + 2v) at v=3 ->", tape.gradient(f, v).numpy())
""")
code("""
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for c, ax in enumerate(axes.ravel()):
    ax.imshow(xtr[np.where(ytr == c)[0][0]]); ax.set_title(C.CLASS_NAMES[c], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
""")
md("### Augmentation (stateless, reproducible)")
code("""
from src.common import augment
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
axes[0].imshow(xtr[0]); axes[0].set_title("original")
for k, ax in enumerate(axes[1:]):
    a, _ = augment(tf.constant(k, tf.int64), tf.constant(xtr[:1]), tf.constant([0]))
    ax.imshow(a[0]); ax.set_title(f"augmented {k+1}")
for ax in axes: ax.axis("off")
plt.show()
""")
md("## 2. Baseline CNN, underfitting and overfitting (Unit II)")
code("""
from src.models import build_baseline
build_baseline().summary()
print("\\nConvolution output size: floor((n + 2p - k) / s) + 1 =", (128 + 2*1 - 3)//1 + 1, "for conv1 (n=128, k=3, p=1, s=1)")
""")
code("""
display(pd.DataFrame({k: {m: v[m] for m in ("params", "epochs_run", "train_acc", "val_acc", "gap")}
                      for k, v in metrics["fit_experiment"]["results"].items()}).T.round(3))
display(Image("assets/plots/overfit_underfit.png")); display(Image("assets/plots/baseline_curves.png"))
""")
md("### A short live training run\nTwo epochs on 480 images, only to show the Keras compile / fit loop running inside the notebook. "
   "The reported models are trained by the scripts.")
code("""
from src.common import seed_everything
seed_everything()
rng = np.random.default_rng(0); pick = rng.choice(len(xtr), 480, replace=False)
demo = build_baseline(img=64)
small = lambda x: np.stack([cv2.resize(i, (64, 64)) for i in x])
import cv2
demo.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
ds, steps = train_dataset(small(xtr[pick]), ytr[pick])
h = demo.fit(ds, steps_per_epoch=steps, epochs=2, validation_data=eval_dataset(small(xva), yva), verbose=2)
""")
md("## 3. Hyperparameter tuning and optimizers (Units I and II)")
code("""
display(pd.DataFrame(metrics["tuning"]["table"]).round(4))
display(pd.DataFrame(metrics["tuning"]["optimizers"]).T.round(4))
display(Image("assets/plots/optimizer_comparison.png"))
""")
md("## 4. Transfer learning with MobileNetV2 (Unit III)\nA pretrained network first classifies a leaf with its original "
   "1000 ImageNet classes, then we use our fine-tuned version.")
code("""
from src.models import build_transfer
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
full = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet", pooling="avg")
print("backbone parameters:", f"{full.count_params():,}", "| feature vector:", full.output.shape)
tran = keras.models.load_model(C.MODELS / "transfer.keras"); base = keras.models.load_model(C.MODELS / "baseline.keras")
t = metrics["transfer"]
print({k: t[k] for k in ("params", "head_params", "finetune_params", "val_acc_after_head", "val_acc_after_ft")})
display(Image("assets/plots/transfer_curves.png"))
""")
md("## 5. Test-set evaluation (recomputed live)")
code("""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
probs = {n: predict(m, xte) for n, m in (("baseline", base), ("transfer", tran))}
for n, p in probs.items():
    print(f"{n}: live accuracy {accuracy_score(yte, p.argmax(1)):.4f} | metrics.json {metrics['evaluation'][n]['accuracy']:.4f}")
print(classification_report(yte, probs["transfer"].argmax(1), target_names=C.CLASS_NAMES, digits=3))
""")
code("""
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for ax, (n, p) in zip(axes, probs.items()):
    sns.heatmap(confusion_matrix(yte, p.argmax(1)), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES if ax is axes[0] else False)
    ax.set(title=n, xlabel="predicted", ylabel="true"); plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
plt.tight_layout(); plt.show()
""")
md("## 6. Save, reload and compare predictions (Unit II)")
code("""
import tempfile
p = Path(tempfile.mkdtemp()) / "copy.keras"
tran.save(p); again = keras.models.load_model(p)
diff = np.abs(predict(again, xte[:32]) - probs["transfer"][:32]).max()
print("largest difference after save -> load:", diff)
""")
md("## 7. Activation maps and Grad-CAM")
code("""
from src.gradcam import activation_maps, make_gradcam, overlay, LAST_CONV
img = to_float(xte[np.where(yte == 2)[0][0]])
for layer, m in activation_maps(base, img).items():
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.6))
    for ax, ch in zip(axes, np.argsort(-m.mean((0, 1)))[:8]):
        ax.imshow(m[..., ch], cmap="viridis"); ax.axis("off")
    fig.suptitle(f"{layer}  {m.shape}", fontsize=9); plt.show()
""")
code("""
cam = make_gradcam(tran, LAST_CONV["transfer"])
yh = probs["transfer"].argmax(1)
idx = list(np.where(yh == yte)[0][[5, 60, 150]]) + list(np.where(yh != yte)[0][:3])
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for j, i in enumerate(idx):
    im = to_float(xte[i]); axes[0, j].imshow(im); axes[1, j].imshow(overlay(im, cam(im)))
    axes[0, j].set_title(f"true: {C.CLASS_NAMES[yte[i]]}", fontsize=7); axes[1, j].set_title(f"pred: {C.CLASS_NAMES[yh[i]]}", fontsize=7)
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout(); plt.show()
""")
md("## 8. Autoencoders (Unit IV)")
code("""
from src.models import encoder_decoder
from src.autoencoder import quality
ae = keras.models.load_model(C.MODELS / "autoencoder.keras"); dae = keras.models.load_model(C.MODELS / "denoiser.keras")
x64 = to_float(load_split("test", 64)[0])
enc, dec = encoder_decoder(ae)
code8 = np.round(enc.predict(x64, verbose=0) * 255).astype(np.uint8)
rec = dec.predict(code8.astype("float32") / 255, verbose=0)
noisy = np.clip(x64 + np.random.default_rng(4).normal(0, C.AE["noise"], x64.shape), 0, 1).astype("float32")
den = dae.predict(noisy, verbose=0)
print("compression", x64[0].size, "B ->", code8[0].size, "B", quality(x64, rec))
print("noisy   ", quality(x64, noisy)); print("denoised", quality(x64, den))
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for r, i in enumerate((10, 250)):
    for ax, im, tt in zip(axes[r], (x64[i], rec[i], noisy[i], den[i]), ("original", "AE reconstruction", "noisy", "denoised")):
        ax.imshow(np.clip(im, 0, 1)); ax.set_title(tt); ax.axis("off")
plt.show()
display(Image("assets/plots/vae_grid.png"))
""")
md("## 9. FGSM adversarial attack and defence (Unit VI)")
code("""
from src.fgsm import robustness
robust = keras.models.load_model(C.MODELS / "baseline_robust.keras")
live = {n: robustness(m, xte, yte) for n, m in (("baseline", base), ("transfer", tran), ("baseline_adv_trained", robust))}
display(pd.DataFrame(live, index=C.FGSM_EPS).round(3).rename_axis("epsilon"))
pd.DataFrame(live, index=C.FGSM_EPS).plot(marker="o", figsize=(7, 4), ylabel="test accuracy"); plt.show()
""")
md("## 10. Background-bias probe\nA random forest that sees only 8 border pixels. Anything above the majority rate means backgrounds leak labels.")
code("""
from sklearn.ensemble import RandomForestClassifier
from src.bias_probe import eight_pixels
rf = RandomForestClassifier(500, random_state=C.SEED, n_jobs=2, class_weight="balanced").fit(eight_pixels(xtr), ytr)
print("8-pixel accuracy", round(rf.score(eight_pixels(xte), yte), 4), "| stored", round(metrics["bias_probe"]["accuracy"], 4))
""")
md("## 11. TensorBoard and the demo app\n- `tensorboard --logdir logs`\n- `streamlit run app/app.py`")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
OUT.parent.mkdir(exist_ok=True)
if "--no-run" not in sys.argv:
    from nbclient import NotebookClient

    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print("wrote", OUT)
