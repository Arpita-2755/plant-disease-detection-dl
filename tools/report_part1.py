"""Report front matter and chapters 1-4."""
from __future__ import annotations

from datetime import date

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

TITLE = ("Plant Leaf Disease Detection using CNN and Transfer Learning, with Explainability, "
         "Autoencoder-based Image Compression and Denoising, and Adversarial Robustness")
REPO = "https://github.com/Arpita-2755/plant-disease-detection-dl"
TODAY = date.today().strftime("%d %B %Y")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def cover(R):
    def line(t, size, bold=False, before=0, after=6, italic=False):
        p = R.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(t)
        r.bold, r.italic, r.font.size = bold, italic, Pt(size)
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
        p.paragraph_format.line_spacing = 1.15

    line("LOVELY PROFESSIONAL UNIVERSITY", 20, True, before=70)
    line("School of Computer Science and Engineering", 15, after=40)
    line("PROJECT REPORT", 18, True, after=10)
    line("on", 12, italic=True, after=10)
    line(TITLE, 17, True, after=14)
    line("deployed as a Streamlit application with Docker", 13, italic=True, after=36)
    line("INT422 – Deep Learning", 15, True, after=2)
    line("(L:2  T:0  P:2, 3 credits)", 12, after=40)
    line("Submitted to", 12, italic=True, after=2)
    line("Professor Anzar Hussain Lone", 14, True, after=28)
    line("Submitted by", 12, italic=True, after=2)
    line("Arpita Mishra", 14, True, after=4)
    line("Registration No.: 12303670", 12, after=2)
    line("Roll No.: 03", 12, after=2)
    line("Section: K2M2359", 12, after=30)
    line(f"Date: {TODAY}", 12)


def front_matter(R, M):
    R.front("Declaration", new_page=False)
    R.p(f"""I, Arpita Mishra, Registration No. 12303670, Roll No. 03, Section K2M2359, student of the School of
        Computer Science and Engineering, Lovely Professional University, declare that the project report titled
        "{TITLE}" is a record of project work submitted in partial fulfilment of the course INT422 – Deep Learning
        under the guidance of Professor Anzar Hussain Lone.""")
    R.p("""The experimental values in this report were produced by the code in the accompanying repository. The dataset,
        pretrained weights, libraries and published work used in the project are acknowledged and cited in the
        References. AI-assisted tooling was used while developing the code and preparing this document.""")
    for t in ("Name: Arpita Mishra", "Registration No.: 12303670", f"Date: {TODAY}", "Signature: ______________________"):
        R.p(t, align="left", after=4)

    R.front("Acknowledgement", new_page=False).paragraph_format.space_before = Pt(36)
    R.ps("""I am thankful to Professor Anzar Hussain Lone for teaching INT422 – Deep Learning and for setting a project
        that asks students to carry an idea all the way from raw data to a deployed application. The practical
        sessions on TensorFlow, convolutional networks, autoencoders and deployment shaped the structure of this work.

        I also thank the School of Computer Science and Engineering, Lovely Professional University, for the
        academic environment in which this project was completed.

        The project depends on openly shared resources. I acknowledge the authors of the PlantVillage dataset, who
        published both the images and the leaf-grouping information that made a fair evaluation possible, and the
        developers of TensorFlow, Keras, Streamlit, scikit-learn and the other open-source libraries used here.""")

    e, a, d, f, b = M["evaluation"], M["autoencoder"], M["denoiser"], M["fgsm"], M["bias_probe"]
    R.front("Abstract")
    R.p(f"""Plant diseases reduce crop yield, and their early signs usually appear on leaves. This project builds an
        end-to-end deep learning system that recognises leaf diseases of tomato, potato and bell pepper, explains its
        decisions, and measures how easily it can be fooled. A reproducible subset of the PlantVillage dataset with
        {M['data']['total']:,} images in ten classes was prepared. Images of the same physical leaf were kept in a single
        split, which avoids a form of data leakage that a random split would introduce.""")
    R.p(f"""Two classifiers were trained on a dual-core CPU. A custom convolutional neural network with
        {M['baseline']['params']:,} parameters reached {pct(e['baseline']['accuracy'])} accuracy on {e['baseline']['total']}
        held-out images. A MobileNetV2 network, pretrained on ImageNet and fine-tuned in two phases after a grid search,
        reached {pct(e['transfer']['accuracy'])} with a macro F1 score of {e['transfer']['macro_f1']:.3f}. Activation maps
        and Grad-CAM heatmaps show which leaf regions influence each prediction.""")
    R.p(f"""A convolutional autoencoder compresses 64x64 leaf images {a['ratio_values']:.0f}:1 at
        {a['uint8_code']['psnr']:.1f} dB PSNR, a denoising variant raises the PSNR of noisy images from
        {d['noisy']['psnr']:.1f} to {d['autoencoder']['psnr']:.1f} dB, and a small variational autoencoder is included for
        comparison. A Fast Gradient Sign Method attack written with `tf.GradientTape` lowers the transfer model's accuracy
        from {pct(f['accuracy']['transfer'][0])} to {pct(f['accuracy']['transfer'][3])} at an epsilon of {f['eps'][3]}, and
        adversarial training is evaluated as a defence. A probe that sees only eight border pixels reaches
        {pct(b['accuracy'])}, which shows that the dataset contains background bias and that laboratory accuracy should not
        be read as field accuracy. The system is delivered as a Streamlit application with a Dockerfile.""")
    R.p("""**Keywords:** convolutional neural network, transfer learning, MobileNetV2, Grad-CAM, autoencoder, FGSM,
        adversarial training, plant disease, TensorFlow, Streamlit.""", align="left")

    h = R.front("Table of Contents", toc=False)
    R.field(R.d.add_paragraph(), ' TOC \\o "1-2" \\h \\z \\u ', "Update field to build the table of contents")
    R.front("List of Figures", toc=False)
    R.field(R.d.add_paragraph(), ' TOC \\h \\z \\c "Figure" ', "Update field")
    R.front("List of Tables", toc=False)
    R.field(R.d.add_paragraph(), ' TOC \\h \\z \\c "Table" ', "Update field")


def chapter1(R, M):
    R.chapter("Introduction", page_break=False)
    R.h2("Background")
    R.ps("""Agriculture depends on keeping crops healthy, and pests and pathogens remain one of the largest causes of lost
        harvest. A global expert survey of five major food crops estimated that pathogens and pests
        cause average yield losses between 17.2% for potato and 30.0% for rice [1]. For a small farmer, the difference between noticing a
        disease in its first week and noticing it in its third week can decide whether a field is saved.

        Many diseases show themselves on leaves first. Early blight produces brown spots with concentric rings, late
        blight produces dark water-soaked patches, leaf mold leaves pale yellow areas, and yellow leaf curl virus bends
        the leaf edges upward. A trained plant pathologist can separate these by eye, but experts are few and cannot visit
        every farm. A phone camera combined with an image classifier offers a first opinion at almost no cost.

        Deep learning suits this task because the visual signs are difficult to describe with hand-written rules. A
        convolutional neural network (CNN) learns its own filters from labelled photographs: edges and colour blobs in the
        early layers, spots and textures in the middle layers, and disease-specific patterns in the last layers [6].
        Transfer learning reduces the amount of labelled data required by reusing filters learned on a much larger
        collection such as ImageNet [13], [14].""")
    R.h2("Problem Statement")
    R.p("""Given a photograph of a single tomato, potato or bell-pepper leaf, the system must decide which one of ten
        conditions it shows, report how confident it is, and show which part of the leaf led to the decision. Beyond
        classification, the project asks four further questions that a working system must face: can leaf images be stored
        in compressed form and cleaned of sensor noise with learned representations; how stable are the predictions when an
        adversary changes pixels slightly; does the measured accuracy reflect leaf disease or some accident of the dataset;
        and can the whole system run on ordinary hardware without a GPU?""")
    R.h2("Motivation")
    R.ps("""Three considerations motivated the choice of topic. The first is practical value, because leaf disease
        recognition is a real need and the output of the model can be understood by a non-specialist. The second is
        coverage of the course. A single dataset carries every practical topic of INT422, from tensors and optimizers to
        convolution, transfer learning, autoencoders, adversarial examples and deployment.

        The third is a gap between headline results and reality. Published work reports accuracy above 99% on PlantVillage
        [2], but the same paper reports 31.4% on images collected under different conditions, and later work showed that a
        model given only eight background pixels can reach 49% on the full dataset [11]. These findings motivated a project
        that treats evaluation as seriously as training.""")
    R.h2("Objectives")
    R.bullets([
        "Prepare a reproducible, leakage-free subset of PlantVillage and study it through exploratory analysis.",
        "Design and train a CNN from scratch, and demonstrate underfitting, overfitting and their remedies.",
        "Apply transfer learning with MobileNetV2 in two phases, with augmentation, learning-rate scheduling, early stopping, checkpointing and TensorBoard logging.",
        "Tune hyperparameters with a search small enough for a CPU.",
        "Evaluate with accuracy, precision, recall, F1 score, confusion matrices and error analysis on unseen leaves.",
        "Explain predictions with activation maps and Grad-CAM.",
        "Build autoencoders for compression and denoising, and compare them with conventional methods.",
        "Implement the FGSM attack from scratch, measure robustness, and test adversarial training.",
        "Deploy the models in a Streamlit application and describe it as a Docker image.",
        "Explain how the workload would scale on an NVIDIA DGX Station A100."])
    R.h2("Scope and Limitations")
    R.p(f"""The project covers three crops and ten classes, with at most 300 images per class, so that all training fits in
        under an hour on a dual-core laptop processor. Images come from a controlled setting with one leaf on a plain
        background. The system is a course demonstration and not a certified diagnostic tool. It cannot recognise a disease
        or crop outside its ten classes, and it will still assign one of those classes to an unrelated picture. No NVIDIA
        GPU was available, so the DGX chapter is a study of published documentation together with a scaling analysis.""")
    R.h2("Real-World Applications")
    R.bullets([
        "First-opinion mobile assistant for farmers and extension workers, with a referral to an expert when confidence is low.",
        "Monitoring in greenhouses, where fixed cameras produce images under controlled light similar to the training data.",
        "Triage for agricultural laboratories that receive many leaf photographs each day.",
        "Compact storage and transmission of leaf images over slow rural networks using a learned code.",
        "Teaching aid that shows students what a CNN looks at and how it can be deceived."])
    R.h2("Mapping to Course Outcomes")
    R.p("""{ref:t-co} relates the six course outcomes to the work in this report. The descriptions are paraphrased by topic
        from the course units that the project was asked to cover.""")
    R.table(["Outcome", "Topic area", "Where it is addressed"], [
        ["CO1", "TensorFlow tensors, computation, optimizers and TensorBoard (Unit I)", "Sections 3.5, 4.1, 6.5, 8.3"],
        ["CO2", "NVIDIA DGX Station A100: hardware, software stack, CUDA toolkit", "Chapter 7"],
        ["CO3", "Keras models, underfitting and overfitting, tuning, saving and loading (Unit II)", "Sections 5.3, 5.5, 8.1 to 8.3, 8.6"],
        ["CO4", "CNN building blocks, transfer learning, augmentation (Unit III)", "Sections 4.2 to 4.4, 5.3, 5.4, 8.2, 8.4, 8.5"],
        ["CO5", "Autoencoders for compression and denoising, VAE (Unit IV)", "Sections 4.5, 5.6, 8.8"],
        ["CO6", "Adversarial robustness, Docker and Streamlit deployment (Unit VI)", "Sections 4.6, 5.7, 6.6, 6.7, 8.9, 8.11"]],
        "Course outcomes and their place in the project", "t-co", [0.8, 3.3, 2.0])
    R.h2("Organisation of the Report")
    R.p("""Chapter 2 reviews published work. Chapter 3 describes the data and Chapter 4 the theory. Chapter 5 presents the
        proposed solution and Chapter 6 its implementation. Chapter 7 covers the NVIDIA DGX Station A100. Chapter 8 reports
        results, Chapter 9 the difficulties met and how they were solved, and Chapter 10 concludes. An appendix contains
        instructions for running the project, key code and the syllabus mapping.""")


def chapter2(R, M):
    R.chapter("Literature Review")
    R.p("""The review follows the order of the project: datasets and plant disease recognition, network architectures,
        training methods, explainability, autoencoders, and adversarial robustness. Each source was checked against
        Crossref, arXiv or the publisher before it was included.""")
    R.h2("Plant Disease Datasets and Recognition")
    R.ps("""Hughes and Salathé [3] released the PlantVillage images to support mobile disease diagnosis. The collection
        contains more than 50,000 expert-labelled photographs of healthy and diseased leaves taken under controlled
        conditions. Mohanty et al. [2] trained AlexNet and GoogLeNet on 54,306 of these images, covering 14 crops and 26
        diseases, and reported 99.35% accuracy for the best configuration. The same study is careful about its limits,
        because accuracy fell to 31.4% on images from other sources. It also recognised that several photographs may show
        one leaf and grouped them when splitting, a precaution that this project adopts.

        Sladojevic et al. [7] built a separate dataset from internet images and fine-tuned a CNN to recognise 13 diseases,
        with reported precision between 91% and 98% per class. Ferentinos [8] trained several CNN architectures on an
        open database of 87,848 images covering 25 plants and 58 plant and disease combinations, and reported a best success
        rate of 99.53%. Too et al. [9] compared the
        fine-tuning of VGG16, Inception V4, ResNet and DenseNet on 38 PlantVillage classes. DenseNet gave the best test accuracy,
        99.75%, with fewer parameters than the others.

        Barbedo [10] studied how dataset size and variety affect such systems and concluded that limited variety is a
        central obstacle even with transfer learning. Noyan [11] went further and trained a model on eight background pixels
        of PlantVillage images. It reached 49.0% on 38 classes, against 2.6% for random guessing, which shows that the
        capture conditions are correlated with the labels. Section 8.10 repeats this test on the subset used here.""")
    R.h2("Convolutional Networks and Transfer Learning")
    R.ps("""LeCun et al. [4] established the modern pattern of convolution, pooling and fully connected layers trained by
        backpropagation. Krizhevsky et al. [5] scaled the idea to ImageNet using GPUs, rectified linear units and dropout,
        and the review by LeCun, Bengio and Hinton [6] explains why learned hierarchical features displaced hand-designed
        ones. Sandler et al. [12] introduced MobileNetV2 for devices with limited computation. Its inverted residual block
        expands a thin tensor, filters it with a depthwise convolution, and projects it back through a linear bottleneck,
        which keeps both parameters and multiplications low. These properties made it the natural choice for a CPU-only
        project.

        Pan and Yang [14] give the general framework of transfer learning, in which knowledge from a source task improves a
        target task with fewer labels. The ImageNet database [13] is the usual source for vision. Shorten and Khoshgoftaar
        [15] survey image augmentation and show that geometric and colour transformations act as a regulariser when data is
        limited.""")
    R.h2("Training Methods")
    R.p("""Ioffe and Szegedy [16] proposed batch normalization, which standardises layer inputs using batch statistics during
        training and moving averages during inference. The distinction between the two modes turned out to matter greatly in
        this project (Section 9.3). Kingma and Ba [17] proposed Adam, which adapts the step size of every parameter from
        estimates of the first and second moments of its gradient.""")
    R.h2("Explainability")
    R.p("""Selvaraju et al. [18] introduced Grad-CAM. The gradient of a class score with respect to the last convolutional
        feature maps is averaged over space to weight each map, and the rectified weighted sum gives a coarse heatmap of the
        regions that support the class. The method needs no change to the network and no retraining, so it can be applied to
        both classifiers here.""")
    R.h2("Autoencoders")
    R.p("""Hinton and Salakhutdinov [19] showed that a deep autoencoder can learn low-dimensional codes that reconstruct data
        better than principal component analysis. Vincent et al. [20] trained networks to recover clean inputs from corrupted
        ones, which forces the code to capture structure and not merely copy pixels. Kingma and Welling [21] turned the
        autoencoder into a generative model by learning a distribution over codes and training it with the reparameterisation
        trick. Reconstruction quality is measured here with mean squared error, peak signal-to-noise ratio and the structural
        similarity index of Wang et al. [25].""")
    R.h2("Adversarial Robustness")
    R.p("""Szegedy et al. [22] discovered that tiny, carefully chosen changes can make a network misclassify an image.
        Goodfellow et al. [23] attributed this to the near-linear behaviour of networks in high dimensions and proposed the
        fast gradient sign method (FGSM), which needs a single gradient. They also proposed training on adversarial examples
        as a defence. Madry et al. [24] framed robust training as a min-max problem and showed that training against the
        stronger, iterative projected gradient descent attack gives more reliable robustness than single-step training.""")
    R.h2("Summary and Identified Gap")
    R.table(["Reference", "Contribution", "Relevance to this project"], [
        ["[2] Mohanty et al., 2016", "99.35% on PlantVillage; 31.4% on other images", "Task definition and leaf grouping"],
        ["[8] Ferentinos, 2018", "Several CNNs on 87,848 images and 58 classes", "Benchmark for supervised CNNs"],
        ["[9] Too et al., 2019", "Fine-tuning study of deep networks", "Supports two-phase fine-tuning"],
        ["[11] Noyan, 2022", "49% from eight background pixels", "Reproduced as a bias probe"],
        ["[12] Sandler et al., 2018", "MobileNetV2 inverted residuals", "Backbone of the transfer model"],
        ["[18] Selvaraju et al., 2017", "Grad-CAM", "Explanation method"],
        ["[20] Vincent et al., 2008", "Denoising autoencoder", "Denoising experiment"],
        ["[23] Goodfellow et al., 2015", "FGSM and adversarial training", "Attack and defence"]],
        "Key literature and its use in this project", "t-lit", [1.8, 2.4, 1.9])
    R.p("""Most student and demonstration projects on PlantVillage report one accuracy figure from a random split. The gap
        addressed here is the combination, on modest hardware, of a leaf-grouped split, explanation of individual decisions,
        a measured robustness curve, a bias probe, and a usable application that exposes all of these to the user.""")


def chapter3(R, M):
    D = M["data"]
    c = D["counts"]
    R.chapter("Dataset and Preprocessing")
    R.h2("Source and Subset Selection")
    R.ps(f"""PlantVillage is distributed by its authors under the Creative Commons Attribution-ShareAlike 3.0 licence. The
        version used here is the colour configuration from the authors' Hugging Face repository [35], fixed to one commit so
        that the files cannot change. The full archive is 2.2 GB. Downloading it was unnecessary for a subset, so
        `prepare_data.py` opens the remote ZIP file with HTTP range requests, reads its index, and fetches only the members
        it needs. Every file is checked against the CRC stored in the archive and a SHA-256 prefix stored in
        `data/manifest.csv`.

        Ten classes were selected from three related crops, with the intention of making the task non-trivial. Early blight
        and late blight occur in both potato and tomato, so the model must separate diseases within a crop as well as similar
        diseases across crops. Up to 300 images were sampled per class. Healthy potato has only 152 images in the whole
        dataset, all of which were used, and this mild imbalance is handled with class weights. The subset contains
        {D['total']:,} images.""")
    R.h2("Leaf-Grouped Splitting")
    R.p(f"""PlantVillage includes several photographs of the same leaf taken from slightly different positions. With an
        ordinary random split, near-copies of a test image may appear in the training set and inflate the score. The authors
        provide a file, `leaf-map.json`, that records which images show the same leaf. The preparation script shuffles leaf
        groups with a fixed seed, selects whole groups, and assigns each group to exactly one of the train, validation and
        test splits in the ratio 70:15:15. A check confirms that the maximum number of splits per leaf is
        {D['max_splits_per_leaf']}. {{ref:t-split}} gives the result.""")
    rows = [[k, v["train"], v["val"], v["test"], sum(v.values()), D["leaf_groups"][k]] for k, v in c.items()]
    ps = D["per_split"]
    rows.append(["Total", ps["train"], ps["val"], ps["test"], D["total"], sum(D["leaf_groups"].values())])
    R.table(["Class", "Train", "Val.", "Test", "Total", "Leaf groups"], rows,
            "Images and leaf groups in each class and split", "t-split", [2.6, 0.7, 0.7, 0.7, 0.7, 0.9],
            align="lccccc", bold_rows=(len(rows) - 1,))
    R.p(f"""Two limitations were found during the audit. First, {D['ungrouped_images']} images have no entry in the leaf map
        and were treated as separate leaves, so a few related images may still be split in the three tomato classes where
        this occurs. Second, {D['duplicate_files']} pairs of files are byte-for-byte identical; both pairs lie entirely in the
        training set and cannot affect the evaluation.""")
    R.h2("Exploratory Data Analysis")
    P = "assets/plots/"
    R.figure(P + "eda_class_distribution.png", "Class distribution across the three splits", "f-dist", 5.0)
    R.figure(P + "eda_sample_grid.png", "Two training images from each class", "f-samples", 4.5, max_h=5.2)
    R.p(f"""All {D['total']:,} images are RGB and {D['image_size'][0].replace('x', ' x ')} pixels. The mean file size is
        {D['mean_file_kb']:.1f} KB and the mean pixel value is {D['mean_brightness']:.0f} on a scale of 0 to 255, with mean
        red, green and blue values of {D['mean_rgb_255'][0]:.0f}, {D['mean_rgb_255'][1]:.0f} and {D['mean_rgb_255'][2]:.0f}.
        {{ref:f-samples}} shows that the images share a single leaf, a plain grey or brown background, and a close viewpoint,
        while {{ref:f-stats}} summarises the pixel statistics. The uniform background makes learning easier, but it is also
        the main reason why results on this dataset do not carry over directly to field photographs.""")
    R.figure(P + "eda_image_stats.png", "File size, brightness and colour statistics of the subset", "f-stats", 5.6)
    R.h2("Preprocessing and Augmentation")
    R.p("""Images are decoded with OpenCV, converted from BGR to RGB and resized with area interpolation to 128 x 128 for
        classification and 64 x 64 for the autoencoders. Pixel values are stored as 8-bit integers in memory and converted to
        32-bit floats in [0, 1] inside the `tf.data` pipeline, which keeps memory use low on the 12 GB test machine. The
        MobileNetV2 model rescales to [-1, 1] with a layer inside the network, so both classifiers accept identical input.""")
    R.p("""Augmentation is applied only to training batches: random horizontal and vertical flips, rotation of up to 20
        degrees, zoom between 0.85 and 1.10, shifts of up to 8%, and changes in brightness and contrast. The affine part
        is a single projective transform applied to the whole batch. All random numbers come from stateless operations
        seeded by the global step, so a repeated run produces the same augmented images. {ref:f-aug} shows examples.""")
    R.figure(P + "eda_augmentation.png", "One training image and nine augmented versions", "f-aug", 4.4)
    R.h2("Tensor Representation")
    R.p(f"""In TensorFlow all data is carried by tensors, described by their rank, shape and data type. {{ref:t-tensors}} was
        printed by `src/eda.py` from real project data. A batch of images is a rank-4 tensor in the order batch, height,
        width, channels. The script also checks automatic differentiation: for f(v) = v² + 2v the gradient at v = 3 is
        computed as {D['tensor_demo']['gradient_at_3']:.0f}, which agrees with the analytical value 2v + 2. A matrix product
        of two flattened images with their transpose gives a tensor of shape
        {tuple(D['tensor_demo']['matmul_shape'])}.""")
    R.table(["Tensor", "Rank", "Shape", "Data type", "Elements"],
            [[t["tensor"], t["rank"], t["shape"], t["dtype"], f"{t['elements']:,}"] for t in D["tensor_demo"]["tensors"]],
            "Tensors met in the project", "t-tensors", [1.5, 0.7, 1.7, 1.0, 1.1], align="lcccc")


def chapter4(R, M):
    R.chapter("Theoretical Background")
    R.h2("Tensors, Computation and Optimization in TensorFlow")
    R.ps("""A tensor is a multi-dimensional array with a fixed data type. TensorFlow [26] records operations on tensors so
        that it can differentiate them. In eager mode operations run immediately, and `tf.function` traces a Python function
        into a graph that can be optimised and reused. `tf.GradientTape` records operations during the forward pass and
        applies the chain rule in reverse to obtain gradients. This project uses Keras `fit` for ordinary training and an
        explicit tape for Grad-CAM, FGSM, adversarial training and the variational autoencoder.

        Training minimises a loss L over parameters w. Plain stochastic gradient descent updates w in the direction of the
        negative gradient. Momentum adds a fraction of the previous update, RMSprop and Adagrad scale each coordinate by the
        size of past gradients, and Adam [17] combines momentum with per-parameter scaling:""")
    R.equation("wₜ₊₁ = wₜ − η · m̂ₜ / (√v̂ₜ + ε)", "4.1")
    R.p("""where m̂ and v̂ are the bias-corrected first and second moment estimates and η is the learning rate. The loss for
        classification is categorical cross-entropy, which for a true class c and predicted probability p_c is:""")
    R.equation("L = − w_c · log p_c", "4.2")
    R.p("""Here w_c is the class weight, equal to N / (K · N_c) for N training images, K classes and N_c images in class c.
        The weight of healthy potato is therefore larger than that of the other classes.""")
    R.h2("Building Blocks of a CNN")
    R.p("""A convolution layer slides small learned kernels over its input and produces one feature map per kernel. Because
        the kernel is shared across positions, the number of parameters does not depend on image size. For input size n,
        kernel size k, padding p and stride s, the output size is:""")
    R.equation("o = ⌊(n + 2p − k) / s⌋ + 1", "4.3")
    R.p("""For the first layer of the baseline, n = 128, k = 3, p = 1 and s = 1 give o = 128. The 2 x 2 max pooling with
        stride 2 that follows gives ⌊(128 − 2) / 2⌋ + 1 = 64. {ref:t-conv} works through every block. The number of weights
        in a convolution without bias is k · k · c_in · c_out, so the first layer has 3 · 3 · 3 · 16 = 432.""")
    f = (16, 32, 64, 128)
    rows, n, cin = [], 128, 3
    for i, c in enumerate(f, 1):
        rows.append([f"Block {i}", f"{n} x {n} x {cin}", f"⌊({n}+2−3)/1⌋+1 = {n}", f"⌊({n}−2)/2⌋+1 = {n // 2}",
                     f"{n // 2} x {n // 2} x {c}", f"{9 * cin * c:,}"])
        n, cin = n // 2, c
    R.table(["Layer", "Input", "Convolution size", "After pooling", "Output", "Conv. weights"], rows,
            "Output sizes of the baseline CNN from Equation 4.3", "t-conv", [0.8, 1.1, 1.5, 1.3, 1.1, 0.9], align="cccccc", size=9.5)
    R.ps("""Batch normalization [16] standardises each channel with the mean and variance of the current batch and keeps
        exponential moving averages for use at inference time. The rectified linear unit, max(0, x), adds non-linearity with
        a constant gradient for positive inputs. Max pooling reduces resolution and gives tolerance to small shifts. Global
        average pooling replaces each feature map by its mean, which removes the very large dense layer that flattening would
        need. Dropout sets a random fraction of activations to zero during training. The softmax output turns scores into
        probabilities that sum to one.""")
    R.h2("Underfitting and Overfitting")
    R.p("""A model underfits when it is too simple, or trained too briefly, to represent the training data: training and
        validation accuracy are both low. It overfits when it memorises training examples: training accuracy is high and
        validation accuracy lags behind while validation loss rises. Remedies for overfitting include more data, augmentation,
        dropout, batch normalization, fewer parameters, early stopping and transfer learning. Section 8.1 demonstrates both
        conditions and their remedy with real runs.""")
    R.h2("Transfer Learning and MobileNetV2")
    R.ps("""A network trained on ImageNet has already learned general filters for edges, colours and textures. In *feature
        extraction* these layers are frozen and only a new classification head is trained. In *fine-tuning* some of the top
        layers are then unfrozen and trained with a small learning rate so that they adapt to the new domain without
        destroying what they have learned.

        MobileNetV2 [12] replaces standard convolutions by depthwise separable ones. A standard convolution costs
        h · w · k² · c_in · c_out multiplications, whereas a depthwise convolution followed by a pointwise convolution costs
        h · w · c_in · (k² + c_out), close to k² times less for large c_out. Each inverted residual block expands the channels
        by a factor of six, filters them depthwise, projects them to few channels without a non-linearity, and adds a shortcut
        when shapes match.""")
    R.h2("Autoencoders")
    R.ps("""An autoencoder consists of an encoder z = e(x) and a decoder x̂ = d(z), trained to make x̂ close to x. If z is
        smaller than x, the code is a learned compression. Quality is measured by the mean squared error, the peak
        signal-to-noise ratio and the structural similarity index:""")
    R.equation("MSE = (1/N) Σ (x − x̂)²,     PSNR = 10 · log₁₀(1 / MSE)", "4.4")
    R.ps("""for pixel values in [0, 1]. SSIM [25] compares local luminance, contrast and structure and equals 1 for identical
        images. A denoising autoencoder [20] receives a corrupted input x + n, with n drawn from a Gaussian distribution, and
        is trained to output the clean x.

        A variational autoencoder [21] encodes each image as a mean μ and variance σ² and samples z = μ + σ · ε with ε from a
        standard normal distribution. Its loss adds to the reconstruction error a Kullback-Leibler term that pulls the code
        distribution towards the standard normal, so that new images can be generated by decoding random codes:""")
    R.equation("L = ‖x − x̂‖² + β · ½ Σ (μ² + σ² − log σ² − 1)", "4.5")
    R.h2("Adversarial Examples and FGSM")
    R.p("""The fast gradient sign method [23] asks how each pixel should change, within a budget ε, to raise the loss as much
        as possible under a linear approximation. The answer is to move every pixel by ε in the direction of the sign of the
        loss gradient:""")
    R.equation("x_adv = clip( x + ε · sign(∇ₓ L(w, x, y)), 0, 1 )", "4.6")
    R.p("""The change is at most ε per pixel and can be invisible, yet the effect accumulates over the 49,152 input values of a
        128 x 128 RGB image. Adversarial training adds such examples to the training batches so that the model learns to
        classify them correctly. The attack here is *white-box*, because it uses the gradients of the model under attack.""")
    R.h2("Grad-CAM")
    R.p("""Let Aᵏ be the k-th feature map of the last convolutional layer and s_c the score of class c. Grad-CAM [18] computes
        a weight for every map by averaging gradients over its Z positions, and combines the maps with a rectifier:""")
    R.equation("αₖ = (1/Z) Σᵢⱼ ∂s_c / ∂Aᵏᵢⱼ,     CAM = ReLU( Σₖ αₖ Aᵏ )", "4.7")
    R.p("""The map is resized to the input and shown as an overlay. The implementation here differentiates log p_c, because the
        gradient of p_c itself vanishes when a prediction is close to certain. Resolution is limited by the layer: 4 x 4 for
        MobileNetV2 at this input size and 16 x 16 for the last activation of the baseline.""")
