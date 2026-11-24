"""Model definitions. Every classifier takes RGB images scaled to [0, 1]."""
from __future__ import annotations

from src import config as C

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L


# ------------------------------------------------------------ baseline CNN
def build_baseline(img=C.IMG, filters=C.BASELINE["filters"], dense=C.BASELINE["dense"],
                   dropout=C.BASELINE["dropout"]) -> keras.Model:
    """Conv-BatchNorm-ReLU-MaxPool blocks, global average pooling, dense head.

    BatchNorm momentum is 0.9: with only 63 steps per epoch the Keras default of
    0.99 leaves the moving variance far from its true value and validation
    accuracy collapses to chance (see the report, Challenges).
    """
    inp = keras.Input((img, img, 3), name="image")
    x = inp
    for i, f in enumerate(filters, 1):
        x = L.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = L.BatchNormalization(momentum=0.9, name=f"bn{i}")(x)
        x = L.ReLU(name=f"relu{i}")(x)
        x = L.MaxPooling2D(2, name=f"pool{i}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(dense, activation="relu", name="dense")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    out = L.Dense(C.NUM_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="baseline_cnn")


# ------------------------------------------------------- transfer learning
def build_transfer(dropout: float, img=C.IMG, alpha=C.TRANSFER["alpha"],
                   weights="imagenet") -> keras.Model:
    """MobileNetV2 backbone (frozen) + pooling, dropout and a softmax head.

    The backbone is built on our own input tensor, so the graph stays flat and
    Grad-CAM can reach its last convolutional activation, `out_relu`.
    """
    inp = keras.Input((img, img, 3), name="image")
    x = L.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inp)
    base = keras.applications.MobileNetV2(input_tensor=x, include_top=False,
                                          weights=weights, alpha=alpha)
    base.trainable = False
    x = L.GlobalAveragePooling2D(name="gap")(base.output)
    x = L.Dropout(dropout, name="dropout")(x)
    out = L.Dense(C.NUM_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="mobilenetv2_transfer")


def unfreeze_top(model: keras.Model, first_layer: str) -> int:
    """Unfreeze layers from `first_layer` on; BatchNorm stays in inference mode."""
    on = False
    for layer in model.layers:
        on = on or layer.name == first_layer
        layer.trainable = on and not isinstance(layer, L.BatchNormalization)
    return sum(int(tf.size(w)) for w in model.trainable_weights)


# ----------------------------------------------------------- autoencoders
def build_autoencoder(name="autoencoder", img=C.AE_IMG, latent_ch=16) -> keras.Model:
    """64x64x3 -> 8x8x16 bottleneck -> 64x64x3."""
    inp = keras.Input((img, img, 3), name="image")
    x = L.Conv2D(32, 3, strides=2, padding="same", activation="relu", name="enc1")(inp)
    x = L.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="enc2")(x)
    z = L.Conv2D(latent_ch, 3, strides=2, padding="same", activation="sigmoid", name="code")(x)
    x = L.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec1")(z)
    x = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="dec2")(x)
    x = L.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", name="dec3")(x)
    out = L.Conv2D(3, 3, padding="same", activation="sigmoid", name="image_out")(x)
    return keras.Model(inp, out, name=name)


def build_denoiser(img=C.AE_IMG) -> keras.Model:
    """Denoising autoencoder with skip connections.

    A plain bottleneck must discard fine detail together with the noise; skips let
    the decoder reuse high-resolution features, so only the noise is removed.
    """
    inp = keras.Input((img, img, 3), name="image")
    e1 = L.Conv2D(32, 3, strides=2, padding="same", activation="relu", name="enc1")(inp)
    e2 = L.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="enc2")(e1)
    z = L.Conv2D(64, 3, padding="same", activation="relu", name="code")(e2)
    d1 = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="dec1")(z)
    d1 = L.Concatenate(name="skip1")([d1, e1])
    d2 = L.Conv2DTranspose(24, 3, strides=2, padding="same", activation="relu", name="dec2")(d1)
    d2 = L.Concatenate(name="skip0")([d2, inp])
    d3 = L.Conv2D(16, 3, padding="same", activation="relu", name="dec3")(d2)
    out = L.Conv2D(3, 3, padding="same", activation="sigmoid", name="image_out")(d3)
    return keras.Model(inp, out, name="denoiser")


def encoder_decoder(ae: keras.Model):
    """Split a trained autoencoder into its encoder and decoder halves."""
    enc = keras.Model(ae.input, ae.get_layer("code").output, name="encoder")
    code_in = keras.Input(enc.output.shape[1:])
    x = code_in
    for layer in ("dec1", "dec2", "dec3", "image_out"):
        x = ae.get_layer(layer)(x)
    return enc, keras.Model(code_in, x, name="decoder")


class VAE(keras.Model):
    """Small convolutional variational autoencoder."""

    def __init__(self, latent=C.AE["vae_latent"], beta=C.AE["vae_beta"], img=C.AE_IMG, **kw):
        super().__init__(**kw)
        self.latent, self.beta = latent, beta
        i = keras.Input((img, img, 3))
        x = L.Conv2D(32, 3, 2, "same", activation="relu")(i)
        x = L.Conv2D(64, 3, 2, "same", activation="relu")(x)
        x = L.Conv2D(64, 3, 2, "same", activation="relu")(x)
        x = L.Flatten()(x)
        self.encoder = keras.Model(i, [L.Dense(latent)(x), L.Dense(latent)(x)], name="vae_encoder")
        zi = keras.Input((latent,))
        x = L.Reshape((8, 8, 32))(L.Dense(8 * 8 * 32, activation="relu")(zi))
        x = L.Conv2DTranspose(64, 3, 2, "same", activation="relu")(x)
        x = L.Conv2DTranspose(32, 3, 2, "same", activation="relu")(x)
        x = L.Conv2DTranspose(16, 3, 2, "same", activation="relu")(x)
        self.decoder = keras.Model(zi, L.Conv2D(3, 3, padding="same", activation="sigmoid")(x),
                                   name="vae_decoder")
        self.seed_gen = keras.random.SeedGenerator(C.SEED)
        self.trackers = {k: keras.metrics.Mean(name=k) for k in ("loss", "recon", "kl")}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def call(self, x, training=False):
        return self.decoder(self.encoder(x)[0])

    def _step(self, x):
        mu, logvar = self.encoder(x, training=True)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        z = mu + tf.exp(0.5 * logvar) * keras.random.normal(tf.shape(mu), seed=self.seed_gen)
        xr = self.decoder(z, training=True)
        # Gaussian likelihood: summed squared error over pixels, averaged per image
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - xr), axis=[1, 2, 3]))
        kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=1))
        return recon + self.beta * kl, recon, kl

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            loss, recon, kl = self._step(x)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for k, v in zip(("loss", "recon", "kl"), (loss, recon, kl)):
            self.trackers[k].update_state(v)
        return {k: m.result() for k, m in self.trackers.items()}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        for k, v in zip(("loss", "recon", "kl"), self._step(x)):
            self.trackers[k].update_state(v)
        return {k: m.result() for k, m in self.trackers.items()}
