"""Write README.md and VIVA_NOTES.md with numbers taken from metrics.json.

  python tools/build_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "metrics.json").read_text())
E, B, T, F, A, D, V, BP = (M[k] for k in ("evaluation", "baseline", "transfer", "fgsm", "autoencoder",
                                          "denoiser", "vae", "bias_probe"))
eb, et, ac = E["baseline"], E["transfer"], F["accuracy"]
ib, it, ir = ac["baseline"], ac["transfer"], ac["baseline_adv_trained"]
fit = M["fit_experiment"]["results"]
best = M["tuning"]["best"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


README = f"""# Plant Leaf Disease Detection

**CNN and transfer learning with Grad-CAM explainability, autoencoder compression and denoising, and FGSM adversarial robustness, deployed as a Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.15-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

![Demo](assets/screenshots/demo.gif)

## Overview

Course project for **INT422 - Deep Learning**, Lovely Professional University. The system recognises ten conditions of
tomato, potato and bell-pepper leaves, shows *why* it made a decision, and measures how easily it can be fooled.
Everything was trained on a dual-core CPU.

## Problem

Crop diseases show early signs on leaves, but experts are scarce. Can a photograph give a useful first opinion, can we see
what the model looks at, and how far can its accuracy be trusted?

## Features

- **Two classifiers:** a CNN built from scratch ({B['params']:,} parameters) and a fine-tuned MobileNetV2
- **Leakage-free evaluation:** images of one physical leaf never cross splits
- **Explainability:** activation maps and Grad-CAM
- **Autoencoders:** {A['ratio_values']:.0f}:1 compression, denoising, and a small VAE
- **Adversarial robustness:** FGSM from scratch with `tf.GradientTape`, plus adversarial training
- **Background-bias probe:** how much can eight border pixels reveal?
- **Streamlit demo** with upload, samples, Grad-CAM, FGSM slider and autoencoder views

## Results

Test set: {et['total']} images of unseen leaves.

| Model | Accuracy | 95% interval | Macro F1 | Parameters | File |
|---|---|---|---|---|---|
| Baseline CNN | **{pct(eb['accuracy'], 2)}** | {100 * eb['wilson95'][0]:.1f} to {100 * eb['wilson95'][1]:.1f}% | {eb['macro_f1']:.3f} | {B['params']:,} | {B['save_load']['file_mb']:.1f} MB |
| MobileNetV2 transfer | **{pct(et['accuracy'], 2)}** | {100 * et['wilson95'][0]:.1f} to {100 * et['wilson95'][1]:.1f}% | {et['macro_f1']:.3f} | {T['params']:,} | {T['save_load']['file_mb']:.1f} MB |

| FGSM epsilon | Baseline | Transfer | Baseline + adversarial training |
|---|---|---|---|
""" + "\n".join(f"| {e:g} | {pct(ib[i])} | {pct(it[i])} | {pct(ir[i])} |" for i, e in enumerate(F["eps"])) + f"""

| Autoencoder task (64x64 test images) | PSNR | SSIM | Reference |
|---|---|---|---|
| Compression to {A['code_values']:,} bytes ({A['ratio_values']:.0f}:1) | {A['uint8_code']['psnr']:.2f} dB | {A['uint8_code']['ssim']:.3f} | JPEG, same size: {A['jpeg_same_budget']['psnr']:.2f} dB |
| Denoising (sigma {D['sigma']}) | {D['autoencoder']['psnr']:.2f} dB | {D['autoencoder']['ssim']:.3f} | noisy {D['noisy']['psnr']:.2f} dB, Gaussian filter {D['gaussian3']['psnr']:.2f} dB |

> **Read these numbers with care.** A random forest that sees only 8 border pixels reaches {pct(BP['accuracy'])}
> (majority class {pct(BP['majority_rate'])}), so PlantVillage backgrounds leak label information. Laboratory accuracy is not field accuracy.

![Model comparison](assets/plots/model_comparison.png)

| Grad-CAM | FGSM robustness |
|---|---|
| ![Grad-CAM](assets/plots/gradcam_correct.png) | ![FGSM](assets/plots/fgsm_robustness.png) |

## Screenshots

| Prediction | Grad-CAM |
|---|---|
| ![Prediction](assets/screenshots/02_upload_prediction.png) | ![Grad-CAM](assets/screenshots/04_gradcam.png) |
| **FGSM attack** | **Autoencoder** |
| ![FGSM](assets/screenshots/06_fgsm_attack.png) | ![Autoencoder](assets/screenshots/07_autoencoder.png) |

## Dataset

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) (CC BY-SA 3.0): {M['data']['total']:,} images, 10 classes, at most
300 per class, split {M['data']['per_split']['train']:,} / {M['data']['per_split']['val']} / {M['data']['per_split']['test']} by leaf.
`data/manifest.csv` records every file, leaf, split and checksum. Images are not stored in git; `src.prepare_data` downloads
about 45 MB with HTTP range requests and verifies them.

## Architecture

![System](assets/diagrams/system_architecture.png)
![CNN](assets/diagrams/cnn_architecture.png)
![Transfer](assets/diagrams/transfer_architecture.png)
![Autoencoder](assets/diagrams/autoencoder_architecture.png)

## Installation

Python 3.12 is required.

```bash
git clone https://github.com/Arpita-2755/plant-disease-detection-dl.git
cd plant-disease-detection-dl
python -m venv .venv
.venv\\Scripts\\activate          # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Notes for Windows: keep the folder path short (TensorFlow has long file names) or enable long paths. If Smart App Control is
on, the first imports may report that a file was blocked while Windows checks it; run the command again.

## Usage

Run the demo (trained models are included):

```bash
streamlit run app/app.py
```

Reproduce everything (under an hour of training on a dual-core CPU):

```bash
python -m src.prepare_data      # download and verify the subset
python -m src.run_all           # EDA, training, evaluation, attacks
tensorboard --logdir logs       # training curves and graphs
```

Single steps: `python -m src.eda`, `src.train_baseline`, `src.tune`, `src.train_transfer`, `src.evaluate`, `src.gradcam`,
`src.autoencoder`, `src.fgsm`, `src.bias_probe`,
`src.tb_graph`, `src.diagrams`. Set `PLANT_FAST=1` and `PLANT_OUTPUT_ROOT=some/folder` for a
quick smoke test that does not touch the real results.

The notebook [`notebooks/plant_disease_workflow.ipynb`](notebooks/plant_disease_workflow.ipynb) reproduces the workflow with outputs
(`pip install -r requirements-dev.txt`).

### Docker

```bash
docker build -t plant-disease .
docker run -p 8501:8501 plant-disease
```

Docker was not available on the development machine, so the image has **not** been built or tested yet. The same start command
was tested outside a container. TensorFlow's Linux wheels need a processor with AVX.

## Project structure

```
app/            Streamlit app and sample leaves (from the test split)
assets/         diagrams, plots, screenshots
data/           manifest.csv (images are downloaded)
logs/           TensorBoard logs (generated)
models/         trained .keras models
notebooks/      executed workflow notebook
report/         project report (.docx, .pdf)
src/            data, models, training, evaluation, attacks
tools/          screenshots, notebook, docs and report builders
metrics.json    every measured number
```

## Tech stack

TensorFlow / Keras, NumPy, pandas, scikit-learn, scikit-image, OpenCV, Matplotlib, seaborn, Streamlit, Altair, TensorBoard,
Docker, Playwright, python-docx.

## Syllabus mapping (INT422)

| Topic | Where |
|---|---|
| Unit I: tensors, optimizers, TensorBoard | `src/eda.py`, `src/tune.py`, `logs/` |
| Unit II: Keras, over/underfitting, tuning, save/load | `src/models.py`, `src/train_baseline.py`, `src/tune.py` |
| Unit III: CNN, transfer learning, augmentation | `src/models.py`, `src/train_transfer.py`, `src/common.py` |
| Unit IV: autoencoders, denoising, VAE | `src/autoencoder.py` |
| Unit VI: FGSM, Docker, Streamlit | `src/fgsm.py`, `Dockerfile`, `app/app.py` |
| Practicals: activation maps, evaluation, pretrained models | `src/gradcam.py`, `src/evaluate.py`, notebook |
| CO2: NVIDIA DGX Station A100, CUDA | Report, Chapter 7 |

## Future work

Field photographs and leaf segmentation, all 38 classes on a GPU, PGD adversarial training, confidence calibration with an
abstain option, and TensorFlow Lite for phones.

## Author

**Arpita Mishra** - Reg. No. 12303670, Roll No. 03, Section K2M2359
School of Computer Science and Engineering, Lovely Professional University
Submitted to Professor Anzar Hussain Lone

## License

Code: [MIT](LICENSE). Leaf images: PlantVillage, CC BY-SA 3.0 (Hughes and Salathé, 2015; Mohanty et al., 2016).
"""

QA = [
    ("What problem does the project solve?", f"It classifies a leaf photo into one of ten conditions of tomato, potato and bell pepper, explains the decision, and tests how reliable it is. Best test accuracy: {pct(et['accuracy'])} on {et['total']} images."),
    ("Which dataset did you use and how large is it?", f"A subset of PlantVillage: {M['data']['total']:,} RGB images of 256x256, 10 classes, up to 300 per class. Healthy potato has only 152, so I used class weights."),
    ("How did you split the data and why?", f"By physical leaf, about 70/15/15: {M['data']['per_split']['train']}/{M['data']['per_split']['val']}/{M['data']['per_split']['test']}. PlantVillage has several photos per leaf; a random split would put near-copies in train and test and inflate accuracy."),
    ("What is a tensor? What are rank and shape?", "A multi-dimensional array with one data type. Rank is the number of axes, shape the size of each. An image batch has rank 4: (32, 128, 128, 3)."),
    ("Describe your baseline CNN.", f"Four blocks of Conv 3x3, BatchNorm, ReLU, MaxPool with 16, 32, 64, 128 filters, then global average pooling, Dense 96, dropout 0.4 and softmax. {B['params']:,} parameters, test accuracy {pct(eb['accuracy'])}."),
    ("How do you compute the output size of a convolution?", "floor((n + 2p - k) / s) + 1. For n=128, k=3, p=1, s=1 the output is 128; 2x2 pooling makes it 64."),
    ("Why global average pooling instead of Flatten?", "Flattening 8x8x128 into Dense 96 needs 786,528 weights; after GAP it needs 12,384. Fewer parameters means less overfitting."),
    ("What is transfer learning and how did you apply it?", f"Reusing knowledge from a source task. I froze MobileNetV2 (ImageNet) and trained a new head ({T['head_params']:,} parameters, {pct(T['val_acc_after_head'])} validation), then unfroze from block_13 with learning rate {T['ft_lr']:g} ({pct(T['val_acc_after_ft'])} validation)."),
    ("Why MobileNetV2?", "It uses depthwise separable convolutions and inverted residuals with linear bottlenecks, so it is light enough for a CPU or phone."),
    ("Why keep BatchNorm frozen during fine-tuning?", "Small batches would give noisy statistics and destroy the ImageNet ones."),
    ("How did you demonstrate overfitting and underfitting?", f"Three models on 400 images. Underfit: {pct(fit['underfit']['train_acc'])} train. Overfit: {pct(fit['overfit']['train_acc'])} train but {pct(fit['overfit']['val_acc'])} validation. Regularised: gap reduced to {100 * fit['regularised']['gap']:.0f} points with augmentation, dropout, BatchNorm and early stopping."),
    ("What was the hardest bug?", f"The baseline trained to 76% but validated at 12%. BatchNorm's moving variance (0.533) had not converged to the real one (0.006) because momentum 0.99 is too slow for 63 steps per epoch. I set momentum 0.9 and re-estimated the statistics on clean training images: validation went from {pct(B['val_before_recalibration'][1])} to {pct(B['val_after_recalibration'][1])}."),
    ("Is that recalibration data leakage?", "No. It uses only training images and changes no weights, only the running mean and variance."),
    ("How did you tune hyperparameters?", f"A grid of {len(M['tuning']['table'])} combinations (learning rate, dropout, batch) on cached MobileNetV2 features. Best: lr {best['lr']:g}, dropout {best['dropout']}, batch {int(best['batch'])}, chosen by validation loss."),
    ("Which optimizers did you compare?", "SGD, SGD with momentum, RMSprop, Adagrad and Adam on the same head. Adaptive ones reduce the loss faster; final accuracies are close."),
    ("What augmentation did you use?", "Flips, rotation up to 20 degrees, zoom, shift, brightness and contrast, on training batches only, with stateless seeded ops so runs repeat exactly."),
    ("Is transfer learning really better here?", f"{pct(et['accuracy'])} against {pct(eb['accuracy'])}; McNemar p = {E['mcnemar']['p_value']:.3f}. Unlikely to be chance alone, but the confidence intervals overlap and the test set is small."),
    ("What is Grad-CAM?", "Average the gradient of the class score over each feature map of the last conv layer to get weights, take the weighted sum, apply ReLU and resize. It shows regions that support the prediction."),
    ("Explain your autoencoder and its compression ratio.", f"64x64x3 = 12,288 bytes become an 8x8x16 code stored as 1,024 bytes: {A['ratio_values']:.0f}:1 at {A['uint8_code']['psnr']:.1f} dB. JPEG at the same size gets {A['jpeg_same_budget']['psnr']:.1f} dB, so JPEG is still better."),
    ("How does the denoiser work and is it useful?", f"It sees noisy input (sigma {D['sigma']}) and learns the clean target; skip connections keep detail. PSNR: noisy {D['noisy']['psnr']:.1f}, Gaussian filter {D['gaussian3']['psnr']:.1f}, denoiser {D['autoencoder']['psnr']:.1f} dB."),
    ("How is a VAE different from an autoencoder?", "The encoder outputs a mean and variance, a code is sampled (reparameterisation trick), and a KL term keeps codes near a standard normal, so random codes can be decoded into new images."),
    ("What is FGSM?", f"x_adv = clip(x + eps * sign(dL/dx), 0, 1). It needs one gradient. At eps = 0.01 the transfer model drops from {pct(it[0])} to {pct(it[3])}."),
    ("How did you defend against it and what did it cost?", f"Adversarial training of the baseline: accuracy at eps 0.01 rises from {pct(ib[3])} to {pct(ir[3])}, clean accuracy falls from {pct(ib[0])} to {pct(ir[0])}. It is not proven against stronger attacks such as PGD."),
    ("Would it work in a real field?", f"Not as it is. Images are from a laboratory, and a model that sees only 8 background pixels gets {pct(BP['accuracy'])}, so the dataset is biased. Field data and segmentation are needed."),
    ("What is the DGX Station A100 and how would you use it?", "A workstation with four A100 GPUs (up to 320 GB), a 64-core EPYC CPU, 512 GB RAM, NVLink and DGX OS with CUDA, cuDNN and containers. I would use MirroredStrategy for data parallelism with NCCL all-reduce and mixed precision; for this small model one MIG slice would be enough."),
]
VIVA = "# Viva notes\n\n25 likely questions with short answers. Numbers come from `metrics.json`.\n\n" + "\n\n".join(
    f"**{i}. {q}**\n{a}" for i, (q, a) in enumerate(QA, 1)) + "\n"

assert len(QA) == 25
(ROOT / "README.md").write_text(README, encoding="utf8")
(ROOT / "VIVA_NOTES.md").write_text(VIVA, encoding="utf8")
print("wrote README.md and VIVA_NOTES.md")
