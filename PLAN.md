# Project plan

Plant leaf disease detection (INT422 Deep Learning). Status as of the final submission.

| Phase | Task | Status |
|---|---|---|
| 1 | Environment, folder structure, pinned requirements | Done |
| 2 | PlantVillage subset, leaf-grouped split, EDA, tensor demonstration | Done |
| 3.1 | Baseline CNN, underfit / overfit / regularised comparison | Done |
| 3.2 | MobileNetV2 feature extraction and fine-tuning | Done |
| 3.3 | Grid search and optimizer comparison | Done |
| 3.4 | Test evaluation, confusion matrices, error analysis | Done |
| 3.5 | Activation maps and Grad-CAM | Done |
| 3.6 | Compression autoencoder, denoiser, VAE | Done |
| 3.7 | FGSM attack and adversarial training | Done |
| 3.8 | Save / load check, metrics.json, TensorBoard | Done |
| 3.9 | Extra: eight-pixel background-bias probe | Done |
| 4 | Streamlit app, real screenshots, demo GIF | Done |
| 4 | Dockerfile (written; not built, Docker is unavailable on this machine) | Done with limitation |
| 5 | Architecture, flow and DGX diagrams | Done |
| 6 | Repository with logical commits | Done |
| 6 | Push to GitHub | Done |
| 7 | Report in .docx and .pdf | Done |
| 8 | Clean-venv verification, number cross-check, checklist, viva notes | Done |

## Decisions

- **Subset:** 10 classes, up to 300 images each, 128x128, because training runs on a 2-core CPU without AVX.
- **Split:** by physical leaf, using the dataset authors' leaf map, to avoid leakage.
- **Download:** HTTP range requests into the authors' Hugging Face ZIP; `raw.githubusercontent.com` was unreachable.
- **Reproducibility:** fixed seeds, deterministic ops, stateless augmentation, pinned dataset commit and checksums.
- **Selection:** every choice uses the validation split; the test split is used once for final figures.
