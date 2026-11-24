"""Convolutional autoencoders: compression, denoising and a small VAE.

  python -m src.autoencoder
"""
from __future__ import annotations

import io
import math

from src import config as C
from src.common import (BLUE, ORANGE, EpochLogger, image_grid, load_split, savefig,
                        seed_everything, timed, to_float, update_metrics)
from src.models import VAE, build_autoencoder, build_denoiser, encoder_decoder

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorflow import keras


def quality(ref: np.ndarray, rec: np.ndarray) -> dict:
    ps = [peak_signal_noise_ratio(a, b, data_range=1.0) for a, b in zip(ref, rec)]
    ss = [structural_similarity(a, b, channel_axis=-1, data_range=1.0) for a, b in zip(ref, rec)]
    return dict(mse=float(np.mean((ref - rec) ** 2)), psnr=float(np.mean(ps)), ssim=float(np.mean(ss)))


def run(model, x, bs=64):
    return np.concatenate([model(x[i:i + bs], training=False).numpy() for i in range(0, len(x), bs)])


def jpeg_at_most(img: np.ndarray, budget: int):
    """Highest-quality JPEG whose size does not exceed `budget` bytes."""
    best = None
    for q in range(1, 96):
        buf = io.BytesIO()
        Image.fromarray((img * 255).round().astype(np.uint8)).save(buf, "JPEG", quality=q)
        if buf.tell() > budget and best is not None:
            break
        best = (np.asarray(Image.open(io.BytesIO(buf.getvalue())), np.float32) / 255, buf.tell())
    return best


def fit(model, xin, xout, vin, vout, name):
    cbs = [keras.callbacks.TensorBoard(str(C.LOGS / name), profile_batch=0),
           keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True),
           keras.callbacks.LearningRateScheduler(lambda ep, lr: C.AE["lr"] * (
               0.05 + 0.475 * (1 + math.cos(math.pi * ep / C.AE["epochs"])))),
           EpochLogger()]
    return model.fit(xin, xout, validation_data=(vin, vout), batch_size=C.AE["batch"], verbose=0,
                     epochs=1 if C.FAST else C.AE["epochs"], callbacks=cbs).history


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    xtr, xva, xte = (to_float(load_split(s, C.AE_IMG)[0]) for s in ("train", "val", "test"))
    sample = rng.choice(len(xte), 6, replace=False)

    # ---------------------------------------------------- 1. compression
    seed_everything()
    ae = build_autoencoder("autoencoder")
    ae.compile(keras.optimizers.Adam(C.AE["lr"]), "mse")
    with timed("autoencoder"):
        h = fit(ae, xtr, xtr, xva, xva, "autoencoder")
    ae.save(C.MODELS / "autoencoder.keras")
    enc, dec = encoder_decoder(ae)
    code = run(enc, xte)
    code8 = np.round(code * 255).astype(np.uint8)      # sigmoid code -> 8-bit integers
    rec = run(dec, code)
    rec8 = run(dec, code8.astype(np.float32) / 255)
    raw_bytes = int(np.prod(xte.shape[1:]))            # 8-bit RGB
    code_bytes = int(np.prod(code8.shape[1:]))
    jp = [jpeg_at_most(im, code_bytes) for im in xte]
    res = dict(
        params=ae.count_params(), input_values=raw_bytes, code_shape=list(code.shape[1:]),
        code_values=code_bytes, ratio_values=raw_bytes / code_bytes,
        ratio_float32_code=raw_bytes / (code_bytes * 4), float_code=quality(xte, rec),
        uint8_code=quality(xte, rec8), jpeg_same_budget=quality(xte, np.stack([j[0] for j in jp])),
        jpeg_mean_bytes=float(np.mean([j[1] for j in jp])), epochs_run=len(h["loss"]),
        history={k: [float(v) for v in vs] for k, vs in h.items()})
    update_metrics("autoencoder", res)
    print({k: v for k, v in res.items() if k != "history"}, flush=True)
    image_grid([xte[i] for i in sample] + [rec8[i] for i in sample] + [jp[i][0] for i in sample],
               ["original"] * 6 + [f"AE {code_bytes} B"] * 6 + [f"JPEG {jp[i][1]} B" for i in sample],
               6, "ae_reconstructions", size=1.6)

    # ------------------------------------------------------ 2. denoising
    sigma = C.AE["noise"]

    def noisy(x, seed):
        g = np.random.default_rng(seed)
        return np.clip(x + g.normal(0, sigma, x.shape), 0, 1).astype(np.float32)

    seed_everything()
    dae = build_denoiser()
    dae.compile(keras.optimizers.Adam(C.AE["lr"]), "mse")
    xn = noisy(xtr, 1)
    with timed("denoiser"):
        h2 = fit(dae, xn, xtr, noisy(xva, 3), xva, "denoiser")
    del xn
    dae.save(C.MODELS / "denoiser.keras")
    nte = noisy(xte, 4)
    drec = run(dae, nte)
    med = np.stack([cv2.medianBlur((im * 255).astype(np.uint8), 3) / 255.0 for im in nte]).astype(np.float32)
    gau = np.stack([cv2.GaussianBlur(im, (3, 3), 0) for im in nte])
    res2 = dict(sigma=sigma, params=dae.count_params(), noisy=quality(xte, nte),
                autoencoder=quality(xte, drec), median3=quality(xte, med), gaussian3=quality(xte, gau),
                epochs_run=len(h2["loss"]), history={k: [float(v) for v in vs] for k, vs in h2.items()})
    update_metrics("denoiser", res2)
    print({k: v for k, v in res2.items() if k != "history"}, flush=True)
    image_grid([xte[i] for i in sample] + [nte[i] for i in sample] + [gau[i] for i in sample]
               + [drec[i] for i in sample],
               ["clean"] * 6 + [f"noisy s={sigma}"] * 6 + ["Gaussian 3x3"] * 6 + ["denoising AE"] * 6,
               6, "denoise_grid", size=1.6)

    # ------------------------------------------------------------ 3. VAE
    seed_everything()
    vae = VAE()
    vae.compile(keras.optimizers.Adam(C.AE["lr"]))
    with timed("vae"):
        h3 = vae.fit(xtr, validation_data=(xva,), batch_size=C.AE["batch"], verbose=0,
                     epochs=1 if C.FAST else C.AE["vae_epochs"], callbacks=[EpochLogger()]).history
    vae.encoder.save(C.MODELS / "vae_encoder.keras")
    vae.decoder.save(C.MODELS / "vae_decoder.keras")
    vrec = run(vae, xte)
    z = rng.normal(size=(12, vae.latent)).astype(np.float32)
    gen = vae.decoder(z).numpy()
    za, zb = (vae.encoder(xte[i:i + 1])[0].numpy() for i in sample[:2])
    path = np.concatenate([za + (zb - za) * t for t in np.linspace(0, 1, 6)])
    image_grid([xte[i] for i in sample] + [vrec[i] for i in sample] + list(gen[:6])
               + list(vae.decoder(path).numpy()),
               ["original"] * 6 + ["VAE recon"] * 6 + ["prior sample"] * 6
               + [f"interp {t:.1f}" for t in np.linspace(0, 1, 6)], 6, "vae_grid", size=1.6)
    res3 = dict(latent=vae.latent, beta=vae.beta, params=vae.encoder.count_params()
                + vae.decoder.count_params(), recon=quality(xte, vrec),
                ratio_values=raw_bytes / vae.latent, epochs_run=len(h3["loss"]),
                final_kl=float(h3["val_kl"][-1]), history={k: [float(v) for v in vs] for k, vs in h3.items()})
    update_metrics("vae", res3)
    print({k: v for k, v in res3.items() if k != "history"}, flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for ax, hh, t, yl in ((axes[0], h, "Compression AE", "MSE"), (axes[1], h2, "Denoising AE", "MSE"),
                          (axes[2], h3, "VAE", "Reconstruction + beta x KL")):
        ep = np.arange(1, len(hh["loss"]) + 1)
        ax.plot(ep, hh["loss"], color=BLUE, label="Training")
        ax.plot(ep, hh["val_loss"], color=ORANGE, label="Validation")
        ax.set(title=t, xlabel="Epoch", ylabel=yl)
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.tight_layout()
    savefig(fig, "ae_curves")


if __name__ == "__main__":
    main()
