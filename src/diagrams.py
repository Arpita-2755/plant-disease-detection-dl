"""Draw every project diagram. Layer names and shapes are read from the models.

  python -m src.diagrams
"""
from __future__ import annotations

from src import config as C
from src.models import build_autoencoder, build_baseline, build_transfer, unfreeze_top

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = C.ASSETS / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})
FILL = dict(blue=("#dcebfb", "#2a78d6"), orange=("#fde5da", "#eb6834"), green=("#d9f2e7", "#1baf7a"),
            yellow=("#fdf0cf", "#c98500"), violet=("#e6e2fb", "#6250d6"), gray=("#f0efec", "#898781"),
            red=("#fbdedd", "#e34948"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def box(ax, x, y, w, h, text, color="blue", fs=9, bold=False):
    f, e = FILL[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
                                fc=f, ec=e, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#0b0b0b",
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p, q, rad=0.0, label=None, dx=0, dy=0.14, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#52514e",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, linestyle=ls))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=8, ha="center",
                color="#52514e")


R = lambda b: (b[0] + b[2], b[1] + b[3] / 2)      # noqa: E731  right edge
Lf = lambda b: (b[0], b[1] + b[3] / 2)            # noqa: E731  left edge
T = lambda b: (b[0] + b[2] / 2, b[1] + b[3])      # noqa: E731  top edge
B = lambda b: (b[0] + b[2] / 2, b[1])             # noqa: E731  bottom edge


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print("saved", name)


def system_architecture():
    fig, ax = canvas(12, 6.4)
    d = box(ax, 0.2, 4.5, 2.0, 1.3, "PlantVillage\n(remote ZIP,\nHTTP range reads)", "gray")
    m = box(ax, 2.8, 4.5, 2.1, 1.3, "prepare_data.py\nleaf-grouped split\nmanifest.csv", "blue")
    p = box(ax, 5.5, 4.5, 2.4, 1.3, "Preprocessing\nresize 128 / 64, scale\nto [0,1], augmentation", "blue")
    arrow(ax, R(d), Lf(m)); arrow(ax, R(m), Lf(p))
    names = [("Autoencoder\nDenoiser, VAE", "violet"), ("Baseline CNN", "orange"),
             ("MobileNetV2\ntransfer", "orange")]
    ms = [box(ax, 0.6 + i * 2.75, 2.45, 2.3, 1.0, n, c, bold=True) for i, (n, c) in enumerate(names)]
    for b in ms:
        arrow(ax, B(p), T(b))
    t = box(ax, 9.0, 2.45, 2.6, 1.0, "tune.py (grid search)\nTensorBoard logs", "yellow")
    arrow(ax, Lf(t), R(ms[2]), ls="--")
    ax.text(8.75, 3.6, "best lr, dropout, batch", fontsize=7.5, ha="center", color="#52514e")
    e = box(ax, 0.6, 0.45, 2.3, 1.15, "evaluate.py\naccuracy, F1,\nconfusion matrix", "green")
    g = box(ax, 3.35, 0.45, 2.3, 1.15, "gradcam.py\nactivation maps,\nheatmaps", "green")
    f = box(ax, 6.1, 0.45, 2.3, 1.15, "fgsm.py\nattack, adversarial\ntraining", "green")
    mj = box(ax, 9.0, 0.45, 2.6, 1.15, "models/*.keras\nmetrics.json\nassets/plots", "gray")
    for a, b in ((ms[1], e), (ms[1], g), (ms[2], e), (ms[2], g), (ms[2], f), (ms[1], f)):
        arrow(ax, B(a), T(b))
    arrow(ax, R(f), Lf(mj))
    s = box(ax, 9.0, 4.5, 2.6, 1.3, "Streamlit app\n(app/app.py)\nDocker, port 8501", "red", bold=True)
    ax.add_patch(FancyArrowPatch((11.6, 1.0), (11.6, 5.1), arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color="#52514e", connectionstyle="arc3,rad=0.16"))
    save(fig, "system_architecture")


def shape_str(layer):
    s = layer.output.shape[1:]
    return " x ".join(str(v) for v in s)


def cnn_architecture():
    m = build_baseline()
    fig, ax = canvas(12, 4.6)
    blocks = [("Input", shape_str(m.layers[0]), "gray")]
    for i, f in enumerate(C.BASELINE["filters"], 1):
        blocks.append((f"Block {i}\nConv2D {f}\n(3x3)\nBatchNorm\nReLU\nMaxPool 2x2",
                       shape_str(m.get_layer(f"pool{i}")), "blue"))
    blocks += [("Global\nAverage\nPooling", shape_str(m.get_layer("gap")), "green"),
               (f"Dense {C.BASELINE['dense']}\nReLU\nDropout {C.BASELINE['dropout']}",
                shape_str(m.get_layer("dense")), "orange"),
               ("Dense\nSoftmax", shape_str(m.get_layer("softmax")), "orange")]
    w, gap = 1.24, 0.26
    prev = None
    for i, (t, s, c) in enumerate(blocks):
        h = 2.6 if c == "blue" else 1.9
        b = box(ax, 0.1 + i * (w + gap), 2.5 - h / 2, w, h, t, c, fs=8)
        ax.text(b[0] + w / 2, 0.75, s, ha="center", fontsize=8, color="#0b0b0b", fontweight="bold")
        if prev:
            arrow(ax, R(prev), Lf(b))
        prev = b
    ax.text(6, 0.35, "output shape (height x width x channels)", ha="center", fontsize=8, color="#52514e")
    ax.text(6, 4.25, f"Baseline CNN: {m.count_params():,} parameters", ha="center", fontsize=12, fontweight="bold")
    save(fig, "cnn_architecture")


def transfer_architecture():
    m = build_transfer(0.2, weights=None)
    total = m.count_params()
    head = m.get_layer("softmax").count_params()
    ft = unfreeze_top(m, C.TRANSFER["unfreeze_from"])
    g = lambda n: shape_str(m.get_layer(n))  # noqa: E731
    rows = [("Input image", g("image"), "gray"), ("Rescaling to [-1, 1]", g("to_minus1_1"), "gray"),
            ("Conv1 32, stride 2", g("Conv1_relu"), "blue"),
            ("Bottleneck, expanded_conv (t=1)", g("expanded_conv_project_BN"), "blue"),
            ("Inverted residuals, blocks 1-2", g("block_2_add"), "blue"),
            ("Inverted residuals, blocks 3-5", g("block_5_add"), "blue"),
            ("Inverted residuals, blocks 6-9", g("block_9_add"), "blue"),
            ("Inverted residuals, blocks 10-12", g("block_12_add"), "blue"),
            ("Inverted residual, block 13", g("block_13_project_BN"), "orange"),
            ("Inverted residuals, blocks 14-15", g("block_15_add"), "orange"),
            ("Inverted residual, block 16", g("block_16_project_BN"), "orange"),
            ("Conv_1 1280 (out_relu)", g("out_relu"), "orange"),
            ("Global average pooling", g("gap"), "green"), ("Dropout (tuned)", g("dropout"), "green"),
            ("Dense softmax", g("softmax"), "green")]
    h = 0.46
    fig, ax = canvas(10, len(rows) * (h + 0.16) + 0.9)
    y0 = len(rows) * (h + 0.16)
    prev, ys = None, []
    for i, (t, s, c) in enumerate(rows):
        y = y0 - i * (h + 0.16)
        b = box(ax, 0.4, y, 4.3, h, t, c, fs=9)
        ax.text(4.9, y + h / 2, s, va="center", fontsize=9, fontweight="bold")
        if prev:
            arrow(ax, B(prev), T(b))
        prev = b
        ys.append(y)

    def brace(i, j, text, color):
        x = 6.6
        ax.plot([x, x], [ys[j] + 0.02, ys[i] + h - 0.02], color=FILL[color][1], lw=3)
        ax.text(x + 0.2, (ys[i] + h + ys[j]) / 2, text, va="center", fontsize=9, linespacing=1.4)

    brace(0, 7, "Frozen ImageNet weights\n(used as fixed feature extractor)", "blue")
    brace(8, 11, f"Unfrozen in phase 2\n(BatchNorm kept frozen)\ntrainable with head: {ft:,}", "orange")
    brace(12, 14, f"New head, trained in phase 1\n{head:,} parameters", "green")
    ax.text(5, y0 + 0.72, f"MobileNetV2 transfer model: {total:,} parameters", ha="center",
            fontsize=12, fontweight="bold")
    save(fig, "transfer_architecture")


def autoencoder_architecture():
    m = build_autoencoder()
    fig, ax = canvas(12, 4.4)
    items = [("Input\nimage", "image", "gray"), ("Conv 32\nstride 2\nReLU", "enc1", "blue"),
             ("Conv 64\nstride 2\nReLU", "enc2", "blue"), ("Code\nConv 16\nstride 2\nsigmoid", "code", "orange"),
             ("ConvT 64\nstride 2\nReLU", "dec1", "green"), ("ConvT 32\nstride 2\nReLU", "dec2", "green"),
             ("ConvT 16\nstride 2\nReLU", "dec3", "green"), ("Conv 3\nsigmoid\noutput", "image_out", "gray")]
    heights = [2.9, 2.3, 1.8, 1.4, 1.8, 2.3, 2.9, 2.9]
    prev = None
    for i, ((t, n, c), h) in enumerate(zip(items, heights)):
        b = box(ax, 0.1 + i * 1.49, 2.3 - h / 2, 1.16, h, t, c, fs=8.5)
        ax.text(b[0] + 0.58, 0.48, shape_str(m.get_layer(n)), ha="center", fontsize=7.6, fontweight="bold")
        if prev:
            arrow(ax, R(prev), Lf(b))
        prev = b
    ax.text(6, 4.1, f"Convolutional autoencoder: {m.count_params():,} parameters, "
            f"{64 * 64 * 3:,} input values to {8 * 8 * 16:,} code values", ha="center", fontsize=11, fontweight="bold")
    ax.text(2.3, 0.1, "encoder", ha="center", fontsize=9, color="#52514e")
    ax.text(8.3, 0.1, "decoder", ha="center", fontsize=9, color="#52514e")
    save(fig, "autoencoder_architecture")


def vertical_flow(steps, name, w=7.2, title=None, side=None):
    h, gap = 0.62, 0.3
    H = len(steps) * (h + gap) + (0.6 if title else 0.1)
    fig, ax = canvas(w, H)
    if title:
        ax.text(w / 2, H - 0.22, title, ha="center", fontsize=12, fontweight="bold")
    prev = None
    boxes = []
    for i, (t, c) in enumerate(steps):
        b = box(ax, 0.6, H - (0.6 if title else 0.1) - (i + 1) * (h + gap) + gap, 4.6 if side else w - 1.2, h, t, c)
        if prev:
            arrow(ax, B(prev), T(b))
        prev = b
        boxes.append(b)
    for (i, j, text) in side or []:
        yi, yj = R(boxes[i])[1], R(boxes[j])[1]
        x = 5.2
        ax.plot([x, x + 0.5, x + 0.5], [yi, yi, yj], color="#52514e", lw=1.4)
        arrow(ax, (x + 0.5, yj), (x, yj))
        ax.text(x + 0.62, (yi + yj) / 2, text, fontsize=8.5, va="center", color="#52514e")
    save(fig, name)


def flows():
    vertical_flow([
        ("Problem definition and syllabus mapping", "gray"),
        ("Download PlantVillage subset (10 classes, 2,852 images)", "blue"),
        ("Leaf-grouped train / validation / test split", "blue"),
        ("EDA and tensor inspection", "blue"),
        ("Baseline CNN, overfit / underfit study", "orange"),
        ("Hyperparameter grid search on MobileNetV2 features", "orange"),
        ("Transfer learning: head training, then fine-tuning", "orange"),
        ("Test evaluation, Grad-CAM, bias probe", "green"),
        ("Autoencoder, denoiser and VAE", "violet"),
        ("FGSM attack and adversarial training", "red"),
        ("Streamlit app and Dockerfile", "yellow"),
        ("Report, README and GitHub repository", "gray")], "project_flowchart",
        title="End-to-end project flow")
    vertical_flow([
        ("Training images (uint8, 128x128x3)", "gray"),
        ("Shuffle with fixed seed, batch", "blue"),
        ("Stateless augmentation: flip, rotate, zoom, shift,\nbrightness, contrast", "blue"),
        ("Forward pass with class-weighted cross-entropy", "orange"),
        ("Backpropagation and Adam update", "orange"),
        ("End of epoch: evaluate on validation split", "green"),
        ("Learning-rate schedule, ModelCheckpoint, TensorBoard", "green"),
        ("EarlyStopping triggered or epochs exhausted?", "yellow"),
        ("Restore best weights, save .keras,\nreload and compare predictions", "violet")],
        "training_pipeline", title="Training pipeline", w=7.6, side=[(7, 1, "no: next\nepoch")])


def fgsm_flow():
    fig, ax = canvas(12, 4.2)
    x = box(ax, 0.1, 2.3, 1.7, 1.1, "Clean image x\n[0, 1], 128x128x3", "gray")
    m = box(ax, 2.4, 2.3, 1.8, 1.1, "Model forward\n(training=False)", "blue")
    l = box(ax, 4.8, 2.3, 1.9, 1.1, "Cross-entropy\nloss J(x, y)", "orange")
    g = box(ax, 7.3, 2.3, 2.1, 1.1, "tf.GradientTape\ngradient dJ/dx", "orange")
    s = box(ax, 10.0, 2.3, 1.8, 1.1, "sign(gradient)", "red")
    for a, b in ((x, m), (m, l), (l, g), (g, s)):
        arrow(ax, R(a), Lf(b))
    y = box(ax, 4.8, 3.75, 1.9, 0.4, "true label y", "gray", fs=8)
    arrow(ax, B(y), T(l))
    a = box(ax, 6.2, 0.4, 5.6, 0.95, "x_adv = clip(x + epsilon * sign(dJ/dx), 0, 1)", "red", bold=True)
    arrow(ax, B(s), (10.9, 1.35), label="times epsilon", dx=0.62, dy=0)
    q = box(ax, 2.4, 0.4, 2.9, 0.95, "Model predicts on x_adv\naccuracy vs. epsilon", "green")
    arrow(ax, Lf(a), R(q))
    ax.add_patch(FancyArrowPatch((0.95, 2.3), (6.2, 1.3), arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 ls="--", color="#898781", connectionstyle="arc3,rad=-0.10"))
    ax.text(1.9, 1.62, "original pixels", fontsize=8, color="#52514e")
    save(fig, "fgsm_flow")


def dgx_scaling():
    fig, ax = canvas(11, 5.2)
    d = box(ax, 3.0, 4.3, 5.0, 0.7, "tf.data pipeline (64-core EPYC host, NVMe cache)", "gray")
    gp = []
    for i in range(4):
        b = box(ax, 0.3 + i * 2.7, 2.3, 2.3, 1.2, f"GPU {i}: A100\nmodel replica\nglobal batch / 4", "blue")
        arrow(ax, B(d), T(b))
        gp.append(b)
    n = box(ax, 0.3, 1.0, 10.4, 0.7, "NVLink all-reduce of gradients (NCCL, MirroredStrategy)", "orange", bold=True)
    for b in gp:
        ax.add_patch(FancyArrowPatch(B(b), (b[0] + b[2] / 2, 1.7), arrowstyle="<|-|>", mutation_scale=13,
                                     lw=1.4, color="#52514e"))
    u = box(ax, 3.1, 0.05, 4.8, 0.6, "identical weight update on every replica", "green")
    arrow(ax, (5.5, 1.0), T(u))
    save(fig, "dgx_scaling")


if __name__ == "__main__":
    system_architecture()
    cnn_architecture()
    transfer_architecture()
    autoencoder_architecture()
    flows()
    fgsm_flow()
    dgx_scaling()
