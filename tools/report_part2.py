"""Report chapters 5-7."""
from __future__ import annotations

import sys
from pathlib import Path

from report_lib import extract
from report_part1 import pct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

DG = "assets/diagrams/"


def chapter5(R, M):
    b, t, a, d, v = M["baseline"], M["transfer"], M["autoencoder"], M["denoiser"], M["vae"]
    R.chapter("Proposed Solution and Methodology")
    R.h2("System Architecture")
    R.p("""The system is organised as independent scripts that communicate through files. {ref:f-sys} shows the flow. Data
        preparation produces a manifest and image folders. Training scripts read them and write models in Keras format,
        TensorBoard logs, plots and a single `metrics.json`. Evaluation, explanation and attack scripts read the saved models.
        The Streamlit application loads the same models and reuses the same Grad-CAM and FGSM functions, so that what is
        demonstrated is exactly what was evaluated.""")
    R.figure(DG + "system_architecture.png", "System architecture", "f-sys", 5.8)
    R.h2("Project Flow")
    R.p("""{ref:f-train}(a) lists the stages in the order in which they were executed. `python -m src.run_all` performs every
        computational stage after the data has been prepared.""")
    R.h2("Baseline CNN")
    R.p(f"""The baseline, shown in {{ref:f-cnn}}, stacks four blocks of convolution, batch normalization, ReLU and max pooling
        with 16, 32, 64 and 128 filters. Global average pooling feeds a dense layer of {C.BASELINE['dense']} units, dropout of
        {C.BASELINE['dropout']} and a softmax output. The network has {b['params']:,} parameters, small enough for CPU training
        and for quick inference. The use of global average pooling is deliberate. A flattened 8 x 8 x 128 tensor followed by
        the same dense layer would need 786,528 weights in that layer alone, against 12,384 here.""")
    R.figure(DG + "cnn_architecture.png", "Baseline CNN with output shapes", "f-cnn", 6.1)
    R.h2("MobileNetV2 Transfer Model")
    R.p(f"""The transfer model ({{ref:f-tl}}) places a rescaling layer before the MobileNetV2 backbone and adds global average
        pooling, dropout and a ten-way softmax. Of {t['params']:,} parameters, only the {t['head_params']:,} in the new head
        are trained in phase 1. In phase 2 the layers from `{t['unfreeze_from']}` onwards are unfrozen, which makes
        {t['finetune_params']:,} parameters trainable, and training continues at a learning rate of {t['ft_lr']:g}. Batch
        normalization layers stay frozen throughout, because their ImageNet statistics are more reliable than those estimated
        from small batches.""")
    R.figure(DG + "transfer_architecture.png", "MobileNetV2 transfer model, layer groups and training phases", "f-tl", 4.6, max_h=4.9)
    R.h2("Training Pipeline and Hyperparameters")
    R.p("""Both classifiers share the pipeline in {ref:f-train}(b). The loss is sparse categorical cross-entropy with class
        weights, and the optimizer is Adam. Early stopping monitors validation loss and restores the best weights, and
        ModelCheckpoint saves them. The baseline uses a cosine learning-rate schedule, while the transfer model halves its
        rate when validation loss stops improving. The values in {ref:t-hp} are taken from `src/config.py` and from the
        record of the final run.""")
    R.pair(DG + "project_flowchart.png", DG + "training_pipeline.png", "End-to-end project flowchart and training pipeline",
           "f-train", ("(a) project flow", "(b) training pipeline"), 2.9, border=False)
    R.table(["Hyperparameter", "Baseline CNN", "Transfer, phase 1", "Transfer, phase 2"], [
        ["Input size", "128 x 128 x 3", "128 x 128 x 3", "128 x 128 x 3"],
        ["Optimizer", "Adam", "Adam", "Adam"],
        ["Initial learning rate", f"{C.BASELINE['lr']:g}", f"{t['head_lr']:g}", f"{t['ft_lr']:g}"],
        ["Schedule", "Cosine decay", "Reduce on plateau", "Reduce on plateau"],
        ["Batch size", C.BATCH, t["batch"], t["batch"]],
        ["Dropout", C.BASELINE["dropout"], t["dropout"], t["dropout"]],
        ["Maximum epochs", C.BASELINE["epochs"], C.TRANSFER["head_epochs"], C.TRANSFER["ft_epochs"]],
        ["Epochs run", b["epochs_run"], t["head_epochs_run"], t["ft_epochs_run"]],
        ["Early-stopping patience", 8, 3, 3],
        ["Trainable parameters", f"{b['params'] - 480:,}", f"{t['head_params']:,}", f"{t['finetune_params']:,}"],
        ["Augmentation", "Yes", "Yes", "Yes"],
        ["Class weights", "Yes", "Yes", "Yes"]],
        "Training hyperparameters of the classifiers", "t-hp", [1.9, 1.4, 1.4, 1.4], align="lccc")
    R.p(f"""The grid search covers learning rate ({', '.join(f'{x:g}' for x in C.TUNE['lr'])}), dropout
        ({', '.join(map(str, C.TUNE['dropout']))}) and batch size ({', '.join(map(str, C.TUNE['batch']))}), giving
        {len(M['tuning']['table'])} configurations of {C.TUNE['epochs']} epochs each. The frozen backbone produces the same
        feature vector for a given image in every epoch, so the 1,280-dimensional features are computed once and only the head
        is trained for each configuration. The whole search therefore takes {M['tuning']['train_seconds'] / 60:.1f} minutes.
        Selection uses validation loss, and the test set is never consulted.""")
    R.h2("Autoencoders")
    R.p(f"""{{ref:f-ae}} shows the compression autoencoder. Three strided convolutions reduce a 64 x 64 x 3 image to a code of
        8 x 8 x 16 and three transposed convolutions restore it. The code layer uses a sigmoid, so each of its 1,024 values
        lies in [0, 1] and can be stored as one byte. The ratio of {a['ratio_values']:.0f}:1 stated in this report is
        therefore a ratio of bytes: {a['input_values']:,} bytes of 8-bit RGB against {a['code_values']:,} bytes of code. If the
        code were kept as 32-bit floats, the ratio would be only {a['ratio_float32_code']:.0f}:1. The denoiser
        has a different purpose, so size does not matter and detail does. It stops at 16 x 16 x 64 and adds two skip connections
        that carry the first encoder output and the noisy image itself to the decoder. It receives images with Gaussian noise of
        standard deviation {d['sigma']} and is trained to output the clean image. The variational
        autoencoder has a {v['latent']}-dimensional code and β = {v['beta']}.""")
    R.figure(DG + "autoencoder_architecture.png", "Convolutional autoencoder with output shapes", "f-ae", 6.1)
    R.table(["Setting", "Compression AE", "Denoising AE", "VAE"], [
        ["Parameters", f"{a['params']:,}", f"{d['params']:,}", f"{v['params']:,}"],
        ["Code size", "8 x 8 x 16 = 1,024", "16 x 16 x 64, with skips", v["latent"]],
        ["Loss", "MSE", "MSE against clean image", "Squared error + β KL"],
        ["Epochs run", a["epochs_run"], d["epochs_run"], v["epochs_run"]],
        ["Batch size", C.AE["batch"], C.AE["batch"], C.AE["batch"]],
        ["Adam learning rate", C.AE["lr"], C.AE["lr"], C.AE["lr"]]],
        "Autoencoder settings", "t-ae-hp", [1.5, 1.5, 1.6, 1.5], align="lccc")
    R.h2("FGSM Attack and Adversarial Training")
    R.p(f"""{{ref:f-fgsm}} shows the attack. The sign of the gradient does not depend on ε, so it is computed once per image and
        reused for the {len(C.FGSM_EPS)} tested values of ε between 0 and {max(C.FGSM_EPS)}. Each attack is white-box: gradients
        come from the model under test. For the defence, a copy of the baseline is fine-tuned for two epochs, with half of
        every batch replaced by FGSM examples whose strength is drawn uniformly between 0 and 0.02. The larger transfer model
        is attacked but not adversarially trained, because its cost on a CPU would exceed the time budget.""")
    R.figure(DG + "fgsm_flow.png", "FGSM attack flow", "f-fgsm", 6.1)
    R.h2("Evaluation Protocol")
    R.p("""All choices, including the best epoch and the best grid configuration, use the validation split. The test split of 420 images is not used to select any model or
        setting. Earlier versions of the pipeline were also scored on it while defects were being repaired (Chapter 9), so it
        is not perfectly untouched, but no hyperparameter was chosen from those scores. Accuracy is reported with a 95% Wilson confidence interval, since with
        a few hundred images a difference of one or two percentage points may be chance. Precision, recall and F1 are averaged
        over classes (macro average), so the smallest class counts as much as the others. An exact McNemar test on the images
        that only one model gets right compares the two classifiers.""")


def chapter6(R, M):
    R.chapter("Implementation")
    R.h2("Hardware and Software")
    R.table(["Item", "Details"], [
        ["Processor", "Intel Pentium 5405U, 2 cores and 4 threads at 2.30 GHz, no AVX instructions"],
        ["Memory", "12 GB"], ["GPU", "None used (integrated Intel UHD Graphics 610)"],
        ["Operating system", "Windows 11 Home"], ["Language", "Python 3.12.10"],
        ["Deep learning", "TensorFlow 2.21.0, Keras 3.15.1, TensorBoard 2.21.0"],
        ["Data and metrics", "NumPy, pandas, SciPy, scikit-learn, scikit-image, OpenCV, Pillow"],
        ["Plots", "Matplotlib, seaborn, Altair"], ["Application", "Streamlit 1.67.0, Docker"],
        ["Tools", "Playwright with Google Chrome for screenshots, python-docx and Microsoft Word for this report, Git"]],
        "Technology stack", "t-stack", [1.5, 4.6])
    R.h2("Project Structure")
    R.code("""plant-disease-detection-dl/
|-- app/                     Streamlit application and sample leaves
|-- assets/                  diagrams/, plots/, screenshots/
|-- data/manifest.csv        exact image list, leaf groups, splits and checksums
|-- logs/                    TensorBoard event files (not tracked)
|-- models/                  trained .keras models
|-- notebooks/               executed workflow notebook
|-- report/                  this report (.docx and .pdf)
|-- src/
|   |-- config.py            paths, classes, seeds, hyperparameters
|   |-- prepare_data.py      download and leaf-grouped split
|   |-- eda.py               exploratory analysis and tensor demonstration
|   |-- models.py            CNN, MobileNetV2, autoencoders, VAE
|   |-- train_baseline.py    baseline CNN, overfit / underfit experiment
|   |-- tune.py              grid search and optimizer comparison
|   |-- train_transfer.py    two-phase transfer learning
|   |-- evaluate.py          test metrics and comparison
|   |-- gradcam.py           activation maps and Grad-CAM
|   |-- autoencoder.py       compression, denoising, VAE
|   |-- fgsm.py              attack and adversarial training
|   |-- bias_probe.py        eight-pixel background probe
|   |-- tb_graph.py          operation graphs for TensorBoard
|   `-- run_all.py           complete pipeline
|-- tools/                   screenshots, notebook and report builders
|-- Dockerfile  requirements.txt  README.md  LICENSE  metrics.json""", size=9)
    R.h2("Data Pipeline")
    R.p("""The training pipeline shuffles with a fixed seed, batches, repeats without end and numbers every batch. The number
        becomes the seed of the stateless augmentation, which makes the stream of augmented images identical in every run.""")
    R.code(extract(ROOT / "src/common.py", "train_dataset"), title="src/common.py: training dataset")
    R.h2("Models and Training")
    R.code(extract(ROOT / "src/models.py", "build_transfer"), title="src/models.py: transfer model")
    R.code(extract(ROOT / "src/models.py", "unfreeze_top"), title="src/models.py: selecting layers for fine-tuning")
    R.p("""The backbone is built directly on the project's input tensor. Had it been nested as a single layer, the
        intermediate activation needed by Grad-CAM would have been hidden inside a sub-model.""")
    sl_b, sl_t = M["baseline"]["save_load"], M["transfer"]["save_load"]
    R.p(f"""After training, each model is saved to the native `.keras` format, loaded again and compared on
        {sl_b['images_checked']} test images. The largest absolute difference in any probability was
        {sl_b['max_abs_difference']:g} for the baseline and {sl_t['max_abs_difference']:g} for the transfer model, and all
        predicted labels agreed. The files occupy {sl_b['file_mb']:.1f} MB and {sl_t['file_mb']:.1f} MB.""")
    R.h2("TensorBoard Monitoring")
    R.p("""The TensorBoard callback writes loss, accuracy and learning rate after every epoch. Keras 3 stores only a conceptual
        graph, so `src/tb_graph.py` traces each classifier and writes its operation graph with one scope per layer.
        The figures below were captured from the live dashboard after training. The command `tensorboard --logdir logs`
        reproduces them.""")
    S = "assets/screenshots/"
    R.figure(S + "tensorboard_scalars.png", "TensorBoard: accuracy of all classifier runs (smoothed)", "f-tb1", 5.6, max_h=3.2, border=True)
    R.figure(S + "tensorboard_graph.png", "TensorBoard: operation graph of the baseline CNN", "f-tb2", 5.6, max_h=3.2, border=True)
    R.h2("Streamlit Application")
    R.p("""The application accepts an uploaded image or one of ten samples, which were drawn at random from the test split
        without looking at predictions. Images are cropped to a centred square and resized. Four tabs show the prediction
        with the three most probable classes, the Grad-CAM overlay, an FGSM slider and the autoencoder views. A sidebar
        switches between the transfer model, the baseline and the adversarially trained baseline. Models are cached after the
        first load. Unreadable files produce an error message, and a warning appears when confidence is below 60%. In the FGSM
        tab the true label of an upload is unknown, so the attack moves away from the model's own prediction. Section 8.11
        shows screenshots.""")
    R.h2("Docker Deployment")
    R.p("""The Dockerfile starts from `python:3.12-slim`, installs the pinned requirements, copies only the source, the
        application and the trained models, exposes port 8501 and adds a health check. Dependencies are installed before the
        code is copied so that Docker can reuse that layer when only code changes.""")
    R.code((ROOT / "Dockerfile").read_text(), title="Dockerfile")
    R.code("docker build -t plant-disease .\ndocker run -p 8501:8501 plant-disease", title="Build and run")
    R.p("""Docker is not installed on the development machine, so the image was **not** built or run there. The file was checked
        by review, and the same start command was tested outside a container. This is recorded as a limitation.""")


def chapter7(R, M):
    b, t = M["baseline"], M["transfer"]
    n = M["data"]["per_split"]["train"]
    ms_b = 1000 * b["train_seconds"] / (b["epochs_run"] * n)
    ep_t = t["head_epochs_run"] + t["ft_epochs_run"]
    ms_t = 1000 * t["train_seconds"] / (ep_t * n)
    R.chapter("NVIDIA DGX Station A100 and Scaling")
    R.p("""This chapter addresses CO2. It summarises public NVIDIA documentation and then explains how the present workload
        would run on such a system. No DGX hardware was available, so no performance measurement on a GPU is claimed.""")
    R.h2("Overview")
    R.p("""The DGX Station A100 is a workstation that places data-centre accelerators in an office enclosure. NVIDIA describes
        it as a system with four fully interconnected A100 Tensor Core GPUs, capable of Multi-Instance GPU operation, with up
        to 320 GB of total GPU memory, which can be plugged into a standard power outlet and serve several users at the same
        time [27].""")
    R.h2("Hardware Architecture")
    R.table(["Component", "Specification", "Source"], [
        ["GPU", "4 x NVIDIA A100, 80 GB each (320 GB total) or 40 GB each (160 GB total)", "[27]"],
        ["GPU performance", "2.5 petaFLOPS AI, 5 petaOPS INT8", "[27]"],
        ["CPU", "1 x AMD EPYC 7742, 64 cores, 2.25 GHz base and 3.4 GHz maximum boost", "[27]"],
        ["System memory", "512 GB DDR4 (8 x 64 GB)", "[27]"],
        ["Data storage", "7.68 TB U.2 NVMe drive for cache and data", "[27]"],
        ["OS storage", "1.92 TB boot M.2 NVMe drive", "[27]"],
        ["Display", "Separate DGX Display GPU for graphics output", "[27]"],
        ["Dimensions", "639 mm high, 256 mm wide, 518 mm deep; gross weight 43.1 kg", "[27]"],
        ["Power", "Consumption can reach 1,500 W under heavy load; 100 to 240 V input", "[27]"],
        ["Operating temperature", "10 to 35 °C", "[27]"]],
        "DGX Station A100 hardware specification", "t-dgx", [1.4, 4.0, 0.7])
    R.h3("The A100 GPU")
    R.ps("""The A100 belongs to the Ampere architecture. The 80 GB version has 6,912 CUDA cores and 432 Tensor Cores, HBM2e
        memory with a bandwidth of 2,039 GB/s in the SXM form, and third-generation NVLink with 600 GB/s of GPU-to-GPU
        bandwidth [28]. CUDA cores execute general parallel work. Tensor Cores perform small matrix multiply-accumulate
        operations in one step and in reduced precision, and account for most of the speed in convolution and matrix
        multiplication. High memory bandwidth matters for convolutional networks because feature maps of a large batch must
        be read and written in every layer.

        NVLink lets the GPUs exchange gradients directly, without passing through the host processor. The user guide notes
        that the additional display GPU has no direct NVLink connectivity and is not suited to compute, so users should
        select the A100 devices explicitly, for example with `CUDA_VISIBLE_DEVICES` or through container options [27].""")
    R.h3("Multi-Instance GPU")
    R.p("""Multi-Instance GPU (MIG) divides one A100 into as many as seven isolated instances, each with its own compute and
        memory resources [28], [31]. Four GPUs can therefore offer up to 28 instances. For a class of students, each could
        train a network of this project's size in a private instance, whereas a single research job could use all four GPUs
        together.""")
    R.h2("Software Stack")
    R.p("""The station ships with DGX OS, which is based on Ubuntu Linux and already contains what is needed to download and run
        containers for deep learning frameworks [27]. {ref:t-dgxsw} arranges the stack from the bottom upward.""")
    R.table(["Layer", "Role"], [
        ["DGX OS (Ubuntu based)", "Operating system tuned and tested by NVIDIA for the hardware"],
        ["NVIDIA GPU driver", "Connects the operating system to the A100 GPUs and manages MIG"],
        ["CUDA Toolkit", "Compiler, runtime, libraries and tools for GPU programs [29]"],
        ["cuDNN, cuBLAS, NCCL", "Accelerated deep learning primitives, linear algebra and collective communication [29], [32]"],
        ["Docker and NVIDIA Container Toolkit", "Libraries and utilities to build and run GPU-accelerated containers [30]"],
        ["NGC containers", "Ready-made framework images from the NVIDIA registry"],
        ["TensorFlow and Keras", "The framework used in this project"],
        ["Application", "Training scripts and the Streamlit service"]],
        "Software stack of a DGX system", "t-dgxsw", [2.3, 3.8])
    R.h2("CUDA Toolkit")
    R.p("""CUDA is NVIDIA's platform for general computing on GPUs. A program launches *kernels* that are executed by thousands
        of threads, grouped into blocks and grids. The CUDA Toolkit [29] provides the tools to write, compile, optimise and
        deploy such programs.""")
    R.bullets([
        "**nvcc**, the compiler driver for CUDA C and C++ code.",
        "**Runtime and driver APIs**, which allocate GPU memory, copy data and launch kernels.",
        "**Libraries**, including cuBLAS for dense linear algebra, cuFFT for Fourier transforms and cuSPARSE for sparse matrices.",
        "**cuDNN**, distributed alongside the toolkit, with tuned convolution, pooling, normalization and activation primitives that TensorFlow calls.",
        "**Nsight Systems and Nsight Compute**, profilers for the whole application and for individual kernels, and **CUDA-GDB**, a debugger.",
        "**nvidia-smi**, installed with the driver, which reports utilisation, memory, temperature and MIG state."])
    R.p("""A Keras user seldom writes kernels. When `model.fit` runs on a GPU, TensorFlow assigns each convolution to a cuDNN
        routine, each matrix product to cuBLAS, and each gradient exchange to NCCL. The version of CUDA and cuDNN must match
        the TensorFlow build, and containers remove most of that difficulty. One detail is relevant here: TensorFlow has not
        supported GPUs on native Windows since version 2.11, so a Linux system such as DGX OS is the proper target.""")
    R.h2("How This Project Would Scale")
    R.h3("Measured starting point")
    R.p(f"""On the dual-core CPU the baseline needed {ms_b:.0f} ms per training image and the transfer model {ms_t:.0f} ms on
        average over both phases, including validation after each epoch. The {n:,} training images therefore took about
        {ms_b * n / 1000:.0f} s and {ms_t * n / 1000:.0f} s per epoch. These figures are measured and serve only as the
        reference for the discussion below.""")
    R.h3("Data-parallel training")
    R.p("""The natural way to use four GPUs is synchronous data parallelism with `tf.distribute.MirroredStrategy`. The strategy
        creates one replica of the model per GPU and mirrors every variable. Each replica processes its share of the batch,
        gradients are combined with an all-reduce that uses NCCL by default on GPUs, and every replica applies the same
        update [33]. {ref:f-dgx} shows the arrangement. The change in the code is small:""")
    R.figure(DG + "dgx_scaling.png", "Data-parallel training on four A100 GPUs", "f-dgx", 4.0)
    R.code("""strategy = tf.distribute.MirroredStrategy()          # 4 x A100
keras.mixed_precision.set_global_policy("mixed_float16")
with strategy.scope():
    model = build_transfer(dropout=0.2)
    model.compile(keras.optimizers.Adam(1e-3 * strategy.num_replicas_in_sync), ...)
global_batch = 64 * strategy.num_replicas_in_sync     # 256""", title="Sketch of the multi-GPU version (not run in this project)")
    R.p(f"""With a batch of 64 per GPU the global batch is 256 and one epoch of {n:,} images needs only
        {-(-n // 256)} steps. The learning rate is usually increased with the batch size and warmed up, because fewer and less
        noisy updates are made. Mixed precision stores activations in 16-bit floating point, which NVIDIA GPUs process faster
        than 32-bit, while variables stay in 32-bit; TensorFlow's guide reports improvements of more than three times on modern
        GPUs [34]. The softmax output should be kept in float32 for numerical stability.""")
    R.h3("Limits of the speed-up")
    R.p("""If a fraction s of the work cannot be parallelised, Amdahl's law bounds the speed-up on G devices by
        1 / (s + (1 − s) / G). With s = 0.05, four GPUs give at most 3.5 times, and with s = 0.20 only 2.5 times. For this
        project the input pipeline is the likely serial part: the model is so small that the GPUs would wait for decoded and
        augmented images. The remedies are caching decoded images in the 512 GB of system memory, augmenting on the GPU,
        parallel `map` calls across the 64 CPU cores, and `prefetch`. A workload of 2,852 small images would not justify four
        A100 GPUs, and one MIG instance would be a better match.""")
    R.h3("What the hardware would make possible")
    R.bullets([
        "Training on all 54,306 PlantVillage images and 38 classes at 224 x 224, with several seeds to report variance.",
        "Iterative adversarial training (PGD) of the transfer model, which multiplies the cost of every step by the number of attack iterations.",
        "A broad hyperparameter search with one trial per MIG instance, up to 28 in parallel.",
        "Serving the Streamlit container with the NVIDIA Container Toolkit (`docker run --gpus all`) to many users."])
    R.table(["Aspect", "This project (laptop CPU)", "DGX Station A100"], [
        ["Compute", "2 CPU cores", "4 x A100 (27,648 CUDA cores, 320 GB)"],
        ["Batch size", f"{C.BATCH} (baseline), {t['batch']} (transfer)", "64 or more per GPU"],
        ["Precision", "float32", "Mixed float16"],
        ["Parallelism", "Threads within one process", "Data parallel over NVLink, or MIG"],
        ["Environment", "Python virtual environment", "NGC container on DGX OS"],
        ["Bottleneck", "Convolution arithmetic", "Input pipeline"]],
        "Comparison of the two environments", "t-dgxcmp", [1.5, 2.2, 2.4])
