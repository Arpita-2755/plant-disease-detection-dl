"""Streamlit demo: plant leaf disease detection.

Run from the project root:
    streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402  (sets TF log level before TF import)

import altair as alt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.common import read_metrics  # noqa: E402
from src.fgsm import fgsm_sign  # noqa: E402
from src.gradcam import LAST_CONV, make_gradcam, overlay  # noqa: E402
from src.models import encoder_decoder  # noqa: E402

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="wide")

MODELS = {
    "MobileNetV2 (transfer learning)": ("transfer", "transfer"),
    "Baseline CNN": ("baseline", "baseline"),
    "Baseline CNN + adversarial training": ("baseline_robust", "baseline"),
}
SAMPLES = Path(__file__).parent / "samples"


@st.cache_resource(show_spinner="Loading model ...")
def get_model(name: str):
    return keras.models.load_model(C.MODELS / f"{name}.keras", compile=False)


@st.cache_resource(show_spinner=False)
def get_gradcam(name: str, kind: str):
    return make_gradcam(get_model(name), LAST_CONV[kind])


@st.cache_resource(show_spinner=False)
def get_autoencoder():
    return encoder_decoder(get_model("autoencoder"))


def load_image(data: bytes) -> np.ndarray:
    """Decode, drop alpha, centre-crop to a square. Returns RGB uint8."""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    return np.asarray(im.crop((left, top, left + side, top + side)))


def resize01(img: np.ndarray, size: int) -> np.ndarray:
    inter = cv2.INTER_AREA if img.shape[0] >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=inter).astype(np.float32) / 255.0


def predict(model, x: np.ndarray) -> np.ndarray:
    return model(x[None], training=False).numpy()[0]


def prob_chart(p: np.ndarray, k: int = 3):
    top = np.argsort(-p)[:k]
    df = pd.DataFrame({"class": [C.CLASS_NAMES[i] for i in top], "probability": p[top]})
    base = alt.Chart(df).encode(y=alt.Y("class:N", sort="-x", title=None,
                                        axis=alt.Axis(labelLimit=240)))
    bars = base.mark_bar(color="#2a78d6", cornerRadiusEnd=4, height=22).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1], title="Probability")),
        tooltip=["class", alt.Tooltip("probability:Q", format=".2%")])
    text = base.mark_text(align="left", dx=4).encode(
        x="probability:Q", text=alt.Text("probability:Q", format=".1%"))
    return (bars + text).properties(height=45 * k + 40)


def show(img, caption):
    st.image(np.clip(img, 0, 1), caption=caption, width="stretch")


# ---------------------------------------------------------------- sidebar
metrics = read_metrics()
with st.sidebar:
    st.header("Settings")
    choice = st.radio("Model", list(MODELS), key="model")
    name, kind = MODELS[choice]
    ev = metrics.get("evaluation", {}).get(name)
    if ev:
        st.metric("Test accuracy", f"{ev['accuracy']:.1%}",
                  help=f"{ev['correct']} of {ev['total']} held-out images correct")
    elif name == "baseline_robust" and "fgsm" in metrics:
        st.metric("Test accuracy", f"{metrics['fgsm']['accuracy']['baseline_adv_trained'][0]:.1%}")
    st.divider()
    st.header("About")
    st.markdown(
        "**Plant Leaf Disease Detection using CNN and Transfer Learning**\n\n"
        "INT422 - Deep Learning, Lovely Professional University\n\n"
        "Arpita Mishra · 12303670 · K2M2359 · Roll No. 03\n\n"
        "Submitted to Prof. Anzar Hussain Lone")
    with st.expander("Classes the model knows"):
        st.markdown("\n".join(f"- {c}" for c in C.CLASS_NAMES))
    st.caption("Images: PlantVillage (CC BY-SA 3.0). Course project - not a substitute "
               "for advice from an agronomist.")

# ------------------------------------------------------------------- input
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a photo of a tomato, potato or bell-pepper leaf, or pick a sample. "
         "The samples come from the held-out test set.")

samples = sorted(SAMPLES.glob("*.jpg"))
origin = st.radio("Image source", ["Upload a photo", "Use a sample leaf"], horizontal=True)
data = None
if origin == "Upload a photo":
    up = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        data, source = up.getvalue(), f"Uploaded: {up.name}"
else:
    sample = st.selectbox("Sample leaf", ["(none)"] + [s.stem.replace("_", " ") for s in samples])
    if sample != "(none)":
        data = next(s for s in samples if s.stem.replace("_", " ") == sample).read_bytes()
        source = f"Sample: {sample}"
if data is None:
    st.info("Choose an image to start.")
    st.stop()
try:
    raw = load_image(data)
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please try a JPG or PNG.")
    st.stop()

model = get_model(name)
x = resize01(raw, C.IMG)
p = predict(model, x)
top = int(p.argmax())

tab_pred, tab_cam, tab_adv, tab_ae = st.tabs(
    ["Prediction", "Grad-CAM", "FGSM attack", "Autoencoder"])

# -------------------------------------------------------------- prediction
with tab_pred:
    a, b = st.columns([1, 2])
    with a:
        st.image(raw, caption=source, width="stretch")
    with b:
        st.subheader(C.CLASS_NAMES[top])
        st.metric("Confidence", f"{p[top]:.1%}")
        if p[top] < 0.6:
            st.warning("Low confidence: the model is unsure about this image.")
        st.markdown("**Top-3 probabilities**")
        st.altair_chart(prob_chart(p), width="stretch")
        st.caption("The model must pick one of its 10 classes, so an image of anything else "
                   "will still receive a label.")

# ---------------------------------------------------------------- Grad-CAM
with tab_cam:
    st.write("Warm regions pushed the model towards the selected class.")
    order = list(np.argsort(-p)[:3])
    c1, c2 = st.columns([2, 1])
    with c1:
        target = st.selectbox("Explain class", order,
                              format_func=lambda i: f"{C.CLASS_NAMES[i]} ({p[i]:.1%})")
    with c2:
        alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
    cam = get_gradcam(name, kind)(x, int(target))
    cols = st.columns(3)
    with cols[0]:
        show(x, f"Model input ({C.IMG}x{C.IMG})")
    with cols[1]:
        st.image((cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]),
                 caption=f"Heatmap from layer '{LAST_CONV[kind]}'", width="stretch")
    with cols[2]:
        show(overlay(x, cam, alpha), "Overlay")

# -------------------------------------------------------------------- FGSM
with tab_adv:
    st.write("FGSM nudges every pixel by ±epsilon in the direction that increases the loss. "
             "Because the true label of an upload is unknown, the attack pushes away from the "
             "model's own prediction.")
    eps = st.slider("Epsilon (fraction of the pixel range)", 0.0, 0.08, 0.01, 0.0025, format="%.4f")
    sign = fgsm_sign(model, x[None], np.array([top])).numpy()[0]
    adv = np.clip(x + eps * sign, 0, 1)
    q = predict(model, adv)
    if eps == 0:
        st.info("Epsilon is 0, so the image is unchanged.")
    elif int(q.argmax()) != top:
        st.error(f"Prediction changed: **{C.CLASS_NAMES[top]}** → **{C.CLASS_NAMES[int(q.argmax())]}** "
                 f"with at most {eps * 255:.1f} of 255 grey levels of change per pixel.")
    else:
        st.success(f"Prediction unchanged at epsilon = {eps:.4f}.")
    cols = st.columns(3)
    with cols[0]:
        show(x, f"Clean: {C.CLASS_NAMES[top]} ({p[top]:.1%})")
    with cols[1]:
        show((sign + 1) / 2, "Perturbation direction (sign of gradient)")
    with cols[2]:
        show(adv, f"Adversarial: {C.CLASS_NAMES[int(q.argmax())]} ({q.max():.1%})")
    st.markdown("**Top-3 probabilities after the attack**")
    st.altair_chart(prob_chart(q), width="stretch")

# ------------------------------------------------------------- autoencoder
with tab_ae:
    s = resize01(raw, C.AE_IMG)

    def scores(r, t):
        return (peak_signal_noise_ratio(r, t, data_range=1.0),
                structural_similarity(r, t, channel_axis=-1, data_range=1.0))

    enc, dec = get_autoencoder()
    code = np.round(enc(s[None], training=False).numpy() * 255).astype(np.uint8)
    rec = dec(code.astype(np.float32) / 255, training=False).numpy()[0]
    psnr, ssim = scores(s, rec)
    st.subheader("Compression")
    m = st.columns(4)
    m[0].metric("Original", f"{s.size:,} B")
    m[1].metric("Code", f"{code.size:,} B", help=f"8-bit code of shape {code.shape[1:]}")
    m[2].metric("Ratio", f"{s.size / code.size:.0f} : 1")
    m[3].metric("PSNR / SSIM", f"{psnr:.1f} dB / {ssim:.2f}")
    big = lambda im: cv2.resize(np.clip(im, 0, 1), (256, 256), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    cols = st.columns(3)
    with cols[0]:
        show(big(s), "Original (64x64)")
    with cols[1]:
        tiles = np.concatenate([np.concatenate(list(code[0, ..., r * 4:r * 4 + 4].transpose(2, 0, 1)), 1)
                                for r in range(4)], 0)
        st.image(cv2.resize(tiles, (256, 256), interpolation=cv2.INTER_NEAREST),
                 caption="Code: 16 channels of 8x8", width="stretch")
    with cols[2]:
        show(big(rec), "Reconstruction")

    st.subheader("Denoising")
    sigma = st.slider("Gaussian noise (standard deviation)", 0.0, 0.3, 0.10, 0.02)
    noisy = np.clip(s + np.random.default_rng(0).normal(0, sigma, s.shape), 0, 1).astype(np.float32)
    clean = get_model("denoiser")(noisy[None], training=False).numpy()[0]
    pn, pc = scores(s, noisy), scores(s, clean)
    cols = st.columns(3)
    with cols[0]:
        show(big(s), "Original")
    with cols[1]:
        show(big(noisy), f"Noisy: {pn[0]:.1f} dB / SSIM {pn[1]:.2f}" if sigma else "Noisy (no noise added)")
    with cols[2]:
        show(big(clean), f"Denoised: {pc[0]:.1f} dB / SSIM {pc[1]:.2f}")
    st.caption("The denoiser was trained for a standard deviation of 0.10.")
