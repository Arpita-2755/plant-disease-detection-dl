"""Central configuration: paths, class list, seeds and hyperparameters.

Two environment variables change behaviour without touching code:
  PLANT_OUTPUT_ROOT  write models/logs/assets/metrics under another folder
  PLANT_FAST=1       tiny smoke run (few images, one epoch) to check wiring
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PLANT_OUTPUT_ROOT", ROOT))
FAST = os.environ.get("PLANT_FAST", "0") == "1"

DATA_DIR = ROOT / "data" / "plantvillage"
MANIFEST = ROOT / "data" / "manifest.csv"
MODELS = OUT / "models"
LOGS = OUT / "logs"
ASSETS = OUT / "assets"
PLOTS = ASSETS / "plots"
METRICS = OUT / "metrics.json"
for _p in (MODELS, LOGS, PLOTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG = 128                 # classifier input resolution
AE_IMG = 64               # autoencoder input resolution
BATCH = 32
PER_CLASS = 300           # maximum images sampled per class
SPLIT = (0.70, 0.15, 0.15)

# PlantVillage folder name -> label shown to people
CLASS_MAP = {
    "Pepper,_bell___Bacterial_spot": "Pepper - Bacterial spot",
    "Pepper,_bell___healthy": "Pepper - Healthy",
    "Potato___Early_blight": "Potato - Early blight",
    "Potato___Late_blight": "Potato - Late blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato___Early_blight": "Tomato - Early blight",
    "Tomato___Late_blight": "Tomato - Late blight",
    "Tomato___Leaf_Mold": "Tomato - Leaf mold",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato - Yellow leaf curl virus",
    "Tomato___healthy": "Tomato - Healthy",
}
CLASS_DIRS = list(CLASS_MAP)
CLASS_NAMES = list(CLASS_MAP.values())
NUM_CLASSES = len(CLASS_NAMES)

# --- training hyperparameters -------------------------------------------
BASELINE = dict(filters=(16, 32, 64, 128), dense=96, dropout=0.4, lr=1e-3, epochs=14)
TRANSFER = dict(alpha=1.0, dropout=0.3, head_lr=1e-3, head_epochs=6,
                ft_lr=1e-4, ft_epochs=8, unfreeze_from="block_13_expand")
TUNE = dict(lr=(3e-3, 1e-3, 3e-4), dropout=(0.2, 0.4), batch=(16, 64), epochs=12)
AE = dict(epochs=10, denoise_epochs=5, lr=2e-3, batch=32, noise=0.10, vae_latent=64, vae_epochs=8, vae_beta=0.5)
FGSM_EPS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)   # in [0, 1] pixel units
