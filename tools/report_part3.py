"""Report chapters 8-10, references and appendix."""
from __future__ import annotations

from pathlib import Path

from docx.shared import Pt

from report_lib import extract
from report_part1 import REPO, pct

ROOT = Path(__file__).resolve().parents[1]
P = "assets/plots/"
S = "assets/screenshots/"


def chapter8(R, M):
    e, b, t = M["evaluation"], M["baseline"], M["transfer"]
    eb, et = e["baseline"], e["transfer"]
    R.chapter("Results and Analysis")
    R.p(f"""All values in this chapter come from the stages of `python -m src.run_all`, executed in a fresh virtual environment, and
        are stored in `metrics.json`. Test results refer to {et['total']} images of leaves that were never seen during
        training or model selection.""")

    # 8.1
    f = M["fit_experiment"]["results"]
    R.h2("Underfitting and Overfitting Experiment")
    R.p(f"""Three small networks were trained on {M['fit_experiment']['train_images']} images at 64 x 64, deliberately few so
        that the effects are visible. The first has only {f['underfit']['params']} parameters. The second has
        {f['overfit']['params'] / 1e6:.1f} million and no regularisation. The third has {f['regularised']['params']:,}
        parameters with batch normalization, dropout, augmentation and early stopping. {{ref:t-fit}} and {{ref:f-fit}} give the
        outcome.""")
    R.table(["Model", "Parameters", "Epochs", "Train acc.", "Val. acc.", "Gap", "Val. loss"],
            [[k.capitalize(), f"{v['params']:,}", v["epochs_run"], pct(v["train_acc"]), pct(v["val_acc"]),
              f"{100 * v['gap']:.1f} pts", f"{v['val_loss']:.2f}"] for k, v in f.items()],
            "Underfitting, overfitting and regularisation", "t-fit", [1.2, 1.1, 0.7, 0.9, 0.9, 0.8, 0.8], align="lcccccc")
    R.figure(P + "overfit_underfit.png", "Training and validation curves of the three models", "f-fit", 5.5)
    R.p(f"""The first model stays close to the 10% expected from guessing on both sets, which is underfitting. The second
        reaches {pct(f['overfit']['train_acc'])} on its training images but only {pct(f['overfit']['val_acc'])} on validation,
        and its validation loss rises after about the fourth epoch while training loss approaches zero, the signature of
        overfitting. The regularised model closes the gap from {100 * f['overfit']['gap']:.0f} to
        {100 * f['regularised']['gap']:.0f} percentage points and lowers validation loss from {f['overfit']['val_loss']:.2f} to
        {f['regularised']['val_loss']:.2f} with {f['overfit']['params'] / f['regularised']['params']:.0f} times fewer
        parameters. Its validation accuracy, {pct(f['regularised']['val_acc'])}, is only slightly higher, so the honest reading
        is that regularisation bought lower loss and more economical learning here, while the larger improvement needs
        more data and better features, as the following sections show.""")

    # 8.2
    R.h2("Baseline CNN Training")
    R.figure(P + "baseline_curves.png", "Baseline CNN training and validation curves", "f-bcurve", 6.1)
    bf, af = b["val_before_recalibration"], b["val_after_recalibration"]
    R.p(f"""The baseline trained for {b['epochs_run']} epochs in {b['train_seconds'] / 60:.1f} minutes. Training accuracy rises
        smoothly, whereas validation accuracy swings widely ({{ref:f-bcurve}}). The cause was traced to batch normalization. The
        moving averages are collected from augmented batches, whose activation statistics differ from those of clean
        validation images. After training, the averages were re-estimated from clean *training* images with all weights fixed.
        Validation accuracy rose from {pct(bf[1])} to {pct(af[1])} and validation loss fell from {bf[0]:.2f} to {af[0]:.2f}.
        The curves in the figure are from before this step and therefore understate the final model. Final accuracy on the
        training images is {pct(b['train_acc_final'])}, so the gap to validation is modest.""")

    # 8.3
    tu = M["tuning"]
    R.h2("Hyperparameter Tuning and Optimizers")
    R.table(["Rank", "Learning rate", "Dropout", "Batch", "Val. accuracy", "Val. loss", "Best epoch"],
            [[i + 1, f"{r['lr']:g}", r["dropout"], int(r["batch"]), pct(r["val_acc"]), f"{r['val_loss']:.3f}", int(r["best_epoch"])]
             for i, r in enumerate(tu["table"])],
            "Grid search results, ordered by validation loss", "t-tune", [0.6, 1.1, 0.8, 0.7, 1.1, 0.9, 0.9], align="ccccccc",
            bold_rows=(0,))
    best = tu["best"]
    worst = tu["table"][-1]
    R.p(f"""The best head uses a learning rate of {best['lr']:g}, dropout {best['dropout']} and batch size {int(best['batch'])}.
        The smallest learning rate has not converged within the epoch budget, as the worst validation loss of
        {worst['val_loss']:.3f} and best epochs at the limit show. Among the leading configurations the differences in accuracy
        are under two percentage points, which corresponds to fewer than nine validation images, so the exact ranking should
        not be over-interpreted.""")
    o = tu["optimizers"]
    R.figure(P + "optimizer_comparison.png", "Five optimizers training the same classification head", "f-opt", 5.4)
    R.table(["Optimizer", "Final training loss", "Final val. accuracy", "Best val. accuracy"],
            [[k, f"{v['final_train_loss']:.3f}", pct(v["final_val_acc"]), pct(v["best_val_acc"])] for k, v in o.items()],
            "Optimizer comparison", "t-opt", [1.6, 1.5, 1.5, 1.5], align="lccc")
    R.p("""The adaptive methods reduce the training loss faster than plain SGD, and momentum closes part of the difference.
        Validation accuracies are near each other, which is expected when all optimizers train the same linear head on fixed
        features.""")

    # 8.4
    R.h2("Transfer Learning")
    R.figure(P + "transfer_curves.png", "Transfer learning: head training followed by fine-tuning", "f-tcurve", 6.1)
    R.p(f"""Phase 1 brings the new head to {pct(t['val_acc_after_head'])} validation accuracy in {t['head_epochs_run']} epochs.
        When layers from `{t['unfreeze_from']}` are unfrozen, accuracy first dips while the new features settle, and then
        climbs to {pct(t['val_acc_after_ft'])} for the restored weights, with validation loss falling from
        {t['val_loss_after_head']:.3f} to {t['val_loss_after_ft']:.3f}. Training accuracy finishes at
        {pct(t['history']['accuracy'][-1])}, so mild overfitting begins in the last epochs and restoring the best epoch is
        useful. Both phases together took {t['train_seconds'] / 60:.1f} minutes.""")

    # 8.5
    R.h2("Test-Set Comparison")
    mc = e["mcnemar"]
    ci = lambda x: f"{100 * x['wilson95'][0]:.1f} to {100 * x['wilson95'][1]:.1f}%"  # noqa: E731
    R.table(["Measure", "Baseline CNN", "MobileNetV2 transfer"], [
        ["Test accuracy", f"{pct(eb['accuracy'], 2)} ({eb['correct']}/{eb['total']})", f"{pct(et['accuracy'], 2)} ({et['correct']}/{et['total']})"],
        ["95% Wilson interval", ci(eb), ci(et)],
        ["Macro precision", f"{eb['macro_precision']:.3f}", f"{et['macro_precision']:.3f}"],
        ["Macro recall", f"{eb['macro_recall']:.3f}", f"{et['macro_recall']:.3f}"],
        ["Macro F1", f"{eb['macro_f1']:.3f}", f"{et['macro_f1']:.3f}"],
        ["Mean confidence when correct", f"{eb['mean_conf_correct']:.2f}", f"{et['mean_conf_correct']:.2f}"],
        ["Mean confidence when wrong", f"{eb['mean_conf_wrong']:.2f}", f"{et['mean_conf_wrong']:.2f}"],
        ["Parameters", f"{b['params']:,}", f"{t['params']:,}"],
        ["Saved model size", f"{b['save_load']['file_mb']:.1f} MB", f"{t['save_load']['file_mb']:.1f} MB"],
        ["Training time", f"{b['train_seconds'] / 60:.1f} min", f"{t['train_seconds'] / 60:.1f} min"],
        ["Single-image latency (compiled graph, CPU)", f"{eb['latency_ms']:.0f} ms", f"{et['latency_ms']:.0f} ms"]],
        "Baseline and transfer model on the test set", "t-cmp", [2.7, 1.7, 1.7], align="lcc", bold_rows=(0,))
    R.figure(P + "model_comparison.png", "Summary metrics and per-class F1 of the two models", "f-cmp", 5.9)
    sig = "below" if mc["p_value"] < 0.05 else "above"
    R.p(f"""Transfer learning corrects {mc['only_transfer_correct']} images that the baseline gets wrong, while the baseline
        corrects {mc['only_baseline_correct']} that the transfer model gets wrong; {mc['both_correct']} are right in both and
        {mc['both_wrong']} wrong in both. The exact McNemar test gives p = {mc['p_value']:.3f}, {sig} the usual 0.05 level.
        The improvement of {100 * (et['accuracy'] - eb['accuracy']):.1f} percentage points is therefore
        {'unlikely to be chance alone, although the two confidence intervals overlap and the test set is small' if sig == 'below' else 'not statistically certain with a test set of this size'}.
        The baseline achieves its result with {t['params'] / b['params']:.0f} times fewer parameters and is
        {et['latency_ms'] / eb['latency_ms']:.1f} times faster for one image, which would matter on a phone.""")
    rows = []
    for c, x in et["per_class"].items():
        y = eb["per_class"][c]
        rows.append([c, int(x["support"]), f"{y['precision']:.2f}", f"{y['recall']:.2f}", f"{y['f1-score']:.2f}",
                     f"{x['precision']:.2f}", f"{x['recall']:.2f}", f"{x['f1-score']:.2f}"])
    R.table(["Class", "Images", "Base P", "Base R", "Base F1", "Transfer P", "Transfer R", "Transfer F1"], rows,
            "Precision (P), recall (R) and F1 per class", "t-class", [2.1, 0.6, 0.55, 0.55, 0.6, 0.7, 0.7, 0.75],
            align="lccccccc", size=9)
    bw = sorted(eb["per_class"].items(), key=lambda kv: kv[1]["f1-score"])[:2]
    tw = sorted(et["per_class"].items(), key=lambda kv: kv[1]["f1-score"])[:2]
    R.p(f"""The hardest classes for the baseline are {bw[0][0]} (F1 {bw[0][1]['f1-score']:.2f}) and {bw[1][0]} (F1
        {bw[1][1]['f1-score']:.2f}); for the transfer model they are {tw[0][0]} ({tw[0][1]['f1-score']:.2f}) and {tw[1][0]}
        ({tw[1][1]['f1-score']:.2f}). Healthy leaves and potato early blight are nearly always recognised. Healthy potato has
        only {int(et['per_class']['Potato - Healthy']['support'])} test images, so each mistake there moves its recall by five
        percentage points.""")
    R.figure(P + "confusion_baseline.png", "Confusion matrix of the baseline CNN", "f-cmb", 4.6, max_h=3.9)
    R.figure(P + "confusion_transfer.png", "Confusion matrix of the transfer model", "f-cmt", 4.6, max_h=3.6)

    # 8.6
    R.h2("Error Analysis")
    tc = lambda x: "; ".join(f"{k.replace(' -> ', ' taken for ')} ({v})" for k, v in list(x["top_confusions"].items())[:3])  # noqa: E731
    R.p(f"""The baseline makes {eb['errors']} errors and the transfer model {et['errors']}. The commonest for the baseline are:
        {tc(eb)}. For the transfer model they are: {tc(et)}. Most mistakes are between diseases that a person could also
        confuse, since early blight, late blight and leaf mold all begin as discoloured patches and differ in texture and
        border more than in colour. At 128 x 128 pixels much of that fine texture has already been lost.""")
    R.figure(P + "misclassified_transfer.png", "Misclassified test images with the highest confidence, transfer model", "f-miss", 5.4, max_h=3.0)
    R.p(f"""Confidence separates correct from wrong predictions to some extent: it averages {et['mean_conf_correct']:.2f} when
        correct and {et['mean_conf_wrong']:.2f} when wrong for the transfer model, against {eb['mean_conf_correct']:.2f} and
        {eb['mean_conf_wrong']:.2f} for the baseline. The transfer model is more accurate but also more confident in its
        errors, so a confidence threshold alone is not a sufficient safeguard.""")
    R.p(f"""Saving and loading was verified for both models, with a maximum difference of
        {max(b['save_load']['max_abs_difference'], t['save_load']['max_abs_difference']):g} in predicted probabilities over
        {b['save_load']['images_checked']} images.""")

    # 8.7
    R.h2("Activation Maps and Grad-CAM")
    R.figure(P + "activation_maps.png", "Activation maps of the baseline CNN for a potato early-blight leaf", "f-act", 5.6)
    R.p("""In {ref:f-act} the first block responds to edges, veins and the leaf outline. The second and third blocks respond
        to lesions and to the contrast between leaf and background. The fourth block is coarse, and its channels light up
        over whole regions of damaged tissue. This is the hierarchy described in Chapter 4.""")
    R.figure(P + "gradcam_correct.png", "Grad-CAM of the transfer model on correct predictions", "f-camok", 5.0, max_h=4.0)
    R.figure(P + "gradcam_incorrect.png", "Grad-CAM of the transfer model on incorrect predictions", "f-camno", 5.0, max_h=4.0)
    R.p("""For correct predictions the warm regions generally fall on the leaf, and on spotted or curled areas in particular.
        For errors the attention is often on the leaf as well, so many of these are confusions between similar symptoms and
        not a failure to find the leaf. In some maps the activation spreads beyond the leaf into shadow or background. The
        4 x 4 resolution of the last MobileNetV2 layer explains part of that spread, but not all of it.""")
    R.figure(P + "gradcam_compare.png", "Grad-CAM of both models on the same leaves", "f-camcmp", 3.4, max_h=4.4)
    R.p("""{ref:f-camcmp} compares the models. The transfer model concentrates on one connected region on the leaf. The
        baseline map is finer but more scattered, and for the tomato early-blight leaf in the last row much of it lies on the
        background and the shadow. The baseline in
        fact misclassifies that leaf as tomato late blight with low confidence (0.50), so its scattered attention goes together
        with a wrong answer. The background activation also connects to the bias probe in Section 8.10.""")

    # 8.8
    a, d, v = M["autoencoder"], M["denoiser"], M["vae"]
    R.h2("Autoencoder Results")
    R.table(["Method", "Bytes per image", "MSE", "PSNR (dB)", "SSIM"], [
        ["Autoencoder, float code", f"{a['code_values'] * 4:,}", f"{a['float_code']['mse']:.5f}", f"{a['float_code']['psnr']:.2f}", f"{a['float_code']['ssim']:.3f}"],
        ["Autoencoder, 8-bit code", f"{a['code_values']:,}", f"{a['uint8_code']['mse']:.5f}", f"{a['uint8_code']['psnr']:.2f}", f"{a['uint8_code']['ssim']:.3f}"],
        ["JPEG, same byte budget", f"{a['jpeg_mean_bytes']:,.0f}", f"{a['jpeg_same_budget']['mse']:.5f}", f"{a['jpeg_same_budget']['psnr']:.2f}", f"{a['jpeg_same_budget']['ssim']:.3f}"],
        ["VAE (64 values)", f"{v['latent'] * 4:,} (float)", f"{v['recon']['mse']:.5f}", f"{v['recon']['psnr']:.2f}", f"{v['recon']['ssim']:.3f}"]],
        "Compression of 64 x 64 test images (12,288 bytes raw)", "t-comp", [2.1, 1.3, 0.9, 0.9, 0.9], align="lcccc")
    R.figure(P + "ae_reconstructions.png", "Originals, autoencoder reconstructions and JPEG at about 1 KB", "f-ae-rec", 4.6)
    gap = a["jpeg_same_budget"]["psnr"] - a["uint8_code"]["psnr"]
    R.p(f"""The autoencoder compresses {a['ratio_values']:.0f}:1 and keeps the outline, colour and large lesions but blurs fine
        texture. Rounding the code to 8 bits changes PSNR by less than 0.01 dB, so the byte ratio is genuine. JPEG at the same
        size is better by {gap:.1f} dB. This is an honest limitation, and the reasons are understandable. JPEG includes entropy
        coding and decades of tuning, while this network has {a['params']:,} parameters, was trained for
        {a['train_seconds'] / 60:.1f} minutes and uses no entropy coder. What the autoencoder offers instead is a compact
        learned feature vector and the possibility of training the decoder for a specific purpose.""")
    R.table(["Image", "MSE", "PSNR (dB)", "SSIM"], [
        [f"Noisy input (σ = {d['sigma']})", f"{d['noisy']['mse']:.5f}", f"{d['noisy']['psnr']:.2f}", f"{d['noisy']['ssim']:.3f}"],
        ["Median filter 3 x 3", f"{d['median3']['mse']:.5f}", f"{d['median3']['psnr']:.2f}", f"{d['median3']['ssim']:.3f}"],
        ["Gaussian filter 3 x 3", f"{d['gaussian3']['mse']:.5f}", f"{d['gaussian3']['psnr']:.2f}", f"{d['gaussian3']['ssim']:.3f}"],
        ["Denoising autoencoder", f"{d['autoencoder']['mse']:.5f}", f"{d['autoencoder']['psnr']:.2f}", f"{d['autoencoder']['ssim']:.3f}"]],
        "Denoising of test images", "t-den", [2.4, 1.2, 1.2, 1.2], align="lccc", bold_rows=(3,))
    R.figure(P + "denoise_grid.png", "Clean, noisy, Gaussian-filtered and autoencoder-denoised images", "f-den", 4.6, max_h=3.6)
    dg = d["autoencoder"]["psnr"] - d["gaussian3"]["psnr"]
    R.p(f"""The denoising autoencoder raises PSNR by {d['autoencoder']['psnr'] - d['noisy']['psnr']:.1f} dB over the noisy input
        and {'exceeds' if dg > 0 else 'falls short of'} the Gaussian filter by {abs(dg):.1f} dB, with an SSIM of
        {d['autoencoder']['ssim']:.3f} against {d['gaussian3']['ssim']:.3f}. An earlier version with a plain 8 x 8 bottleneck
        reached only {M['denoiser_v1']['psnr']:.1f} dB, which was worse than the Gaussian filter, and motivated the skip connections (Section 9.6).""")
    R.figure(P + "vae_grid.png", "VAE: originals, reconstructions, random samples and interpolation", "f-vae", 4.4, max_h=3.5)
    R.p(f"""The VAE squeezes an image into {v['latent']} numbers and reaches {v['recon']['psnr']:.1f} dB. Its outputs are blurred,
        a known effect of the Gaussian reconstruction term together with a small code. Random samples show leaf-like shapes on
        a grey background, and interpolation changes shape gradually, which indicates that the code space is smooth.
        Validation loss followed training loss closely for all three autoencoders and was still falling slowly at the end, so
        they underfit slightly within the time budget (curves in `assets/plots/ae_curves.png`).""")

    # 8.9
    F = M["fgsm"]
    R.h2("Adversarial Robustness")
    ac = F["accuracy"]
    R.table(["Epsilon", "Grey levels (of 255)", "Baseline", "Transfer", "Baseline + adv. training"],
            [[f"{ep:g}", f"{ep * 255:.1f}", pct(ac["baseline"][i]), pct(ac["transfer"][i]), pct(ac["baseline_adv_trained"][i])]
             for i, ep in enumerate(F["eps"])], "Test accuracy under white-box FGSM", "t-fgsm", [0.9, 1.3, 1.1, 1.1, 1.7], align="ccccc")
    R.figure(P + "fgsm_robustness.png", "Accuracy against FGSM strength", "f-rob", 4.7)
    R.p(f"""Both standard models are fragile. At ε = {F['eps'][2]}, which is little more than one grey level per pixel, the
        baseline falls to {pct(ac['baseline'][2])} and the transfer model to {pct(ac['transfer'][2])}. At ε = {F['eps'][3]} they
        reach {pct(ac['baseline'][3])} and {pct(ac['transfer'][3])}. The more accurate model is the less robust one here.
        A likely reason is
        that the deeper network gives the attack a longer chain of near-linear operations to exploit, but this was not tested. At ε = {F['eps'][6]} the accuracy of the standard models rises
        slightly. The step
        probably overshoots the region in which the linear approximation holds. Accuracy stays below random guessing and the
        images are visibly corrupted, so this is not a recovery of robustness.""")
    R.figure(P + "fgsm_examples.png", "Clean and adversarial examples for the transfer model (red titles: wrong prediction)", "f-fex", 4.8, max_h=4.2)
    i = 3
    R.p(f"""Adversarial training raises accuracy at ε = {F['eps'][i]} from {pct(ac['baseline'][i])} to
        {pct(ac['baseline_adv_trained'][i])} after only {F['adv_training_seconds'] / 60:.1f} minutes of extra training. The cost
        is a drop in clean accuracy from {pct(ac['baseline'][0])} to {pct(ac['baseline_adv_trained'][0])}, the trade-off usually
        reported in the literature. Robustness to FGSM does not imply robustness to iterative attacks such as projected
        gradient descent [24], which were not tested.""")

    # 8.10
    bp = M["bias_probe"]
    R.h2("Background-Bias Probe")
    R.p(f"""Following Noyan [11], a random forest was trained on the colours of eight border pixels per image: four corners and
        four edge midpoints. These pixels rarely contain leaf tissue. The probe reached {pct(bp['accuracy'])} on the test set,
        where the largest class accounts for {pct(bp['majority_rate'])}. The background therefore carries a great deal of label
        information, probably because images of one class tend to share light, backdrop
        and camera. This does
        not prove that the CNNs rely on the background, and Grad-CAM shows that they mostly attend to the leaf, but it does mean
        that test accuracy on PlantVillage is an optimistic estimate of performance on new photographs. The leaf-grouped split
        removes leakage between copies of the same leaf but cannot remove this capture bias.""")

    # 8.11
    R.h2("Demonstration Application")
    R.p("""The following screenshots were captured automatically from the running application with Playwright. The uploaded
        image is a test image that is not among the bundled samples.""")
    R.figure(S + "02_upload_prediction.png", "Prediction for an uploaded leaf", "f-s2", 5.6, max_h=3.15, border=True)
    R.pair(S + "01_home.png", S + "03_sample_prediction.png", "Application home screen and a bundled sample", "f-s1",
           ("(a) home screen", "(b) sample leaf"))
    R.figure(S + "04_gradcam.png", "Grad-CAM tab", "f-s4", 5.6, max_h=3.15, border=True)
    R.figure(S + "06_fgsm_attack.png", "FGSM tab after the attack (epsilon 0.02)", "f-s6", 5.6, max_h=3.15, border=True)
    R.pair(S + "05_fgsm_eps0.png", S + "09_baseline_model.png", "FGSM tab before the attack, and the baseline model selected",
           "f-s5", ("(a) epsilon 0", "(b) baseline CNN"))
    R.pair(S + "07_autoencoder.png", S + "08_denoiser.png", "Autoencoder tab", "f-s7", ("(a) compression", "(b) denoising"))
    R.p("""Every interaction shown was executed by the script without an exception in the page, and the script fails if the
        application displays an error.""")


def chapter9(R, M):
    q = M["bn_diagnosis"]
    b = M["baseline"]
    R.chapter("Challenges Faced and Solutions")
    R.h2("A Processor without AVX and a Security Policy")
    R.p("""The laptop has a Pentium processor without AVX instructions, which TensorFlow builds normally expect, and Windows
        Smart App Control initially blocked some TensorFlow library files that it had not yet evaluated. No security setting was
        changed. After repeated imports the policy completed its own reputation checks and allowed the files, and TensorFlow
        2.21 was confirmed to train correctly. Long Windows paths caused one failed installation, solved by a shorter location
        for the virtual environment.""")
    R.h2("Reaching the Dataset")
    R.p("""The GitHub raw-file service could not be reached from the network, and the Hugging Face archive is 2.2 GB. Reading
        the ZIP index remotely and fetching single members solved the problem. A first version sent one redirect request per
        file and was rate-limited by the server, so the script now resolves the download address once, reuses it, and waits when
        asked to. An audit then found 153 healthy potato images where the dataset has 152, because directory entries in the ZIP
        file had been counted as images; they are now skipped and the subset was rebuilt.""")
    R.h2("Batch Normalization Failing at Inference")
    R.p(f"""In the first full run the baseline reached {pct(q['train_acc_epoch6'])} training accuracy by the sixth epoch but only
        {pct(q['val_acc_epoch6'])} on validation. Since the transfer model worked with the same data, the pipeline was not at
        fault. Inspection of the saved network showed a moving variance of {q['bn1_moving_var_median']:.3f} in the first batch
        normalization layer against a true batch variance of {q['bn1_batch_var_median']:.4f}. The Keras default momentum of 0.99
        changes the average by only 1% per step, and with 63 steps per epoch the initial value of 1 had not decayed. Refreshing
        the averages of that network, with no change to its weights, lifted validation accuracy from
        {pct(q['val_acc_stale_stats'])} to {pct(q['val_acc_refreshed_stats'])}. The momentum was set to 0.9.""")
    R.p(f"""A second, subtler effect remained: augmented batches have different statistics from clean images. The final step of
        training now re-estimates the averages from clean training images, which raised validation accuracy from
        {pct(b['val_before_recalibration'][1])} to {pct(b['val_after_recalibration'][1])}. This was the most instructive problem
        of the project, because the training curve alone gave no sign of it.""")
    R.h2("Early Stopping on a Noisy Signal")
    R.p("""Because validation loss fluctuated, early stopping ended one run at epoch 8 and restored weights from epoch 3 although
        the model was still improving. A cosine learning-rate schedule and a longer patience replaced the plateau rule for the
        baseline.""")
    R.h2("Limited Computing Time")
    total = sum(M[k].get("train_seconds", 0) for k in ("fit_experiment", "baseline", "tuning", "transfer", "autoencoder", "denoiser", "vae")) \
        + M["fgsm"]["adv_training_seconds"]
    R.p(f"""All training had to fit on two CPU cores. The measures taken were a 128 x 128 input, a compact baseline, MobileNetV2,
        cached features for the grid search, 64 x 64 autoencoders and two epochs of adversarial training. Total training time in
        the final run was {total / 60:.0f} minutes.""")
    R.h2("A Denoiser Weaker than a Simple Filter")
    R.p(f"""The first denoising autoencoder copied the compression architecture and reached
        {M['denoiser_v1']['psnr']:.1f} dB, below the {M['denoiser_v1']['gaussian3_psnr']:.1f} dB of a
        3 x 3 Gaussian filter. Its bottleneck discarded fine detail along with the noise. Skip connections from the encoder and
        from the input allow the decoder to reuse high-resolution information. Comparing with a simple baseline was what exposed
        the problem, and it would have been missed had only the improvement over the noisy input been reported. The comparison was
        made on test images, so the redesign was informed by them. It was a single change of architecture, with no further
        tuning afterwards.""")
    R.h2("Reproducibility")
    R.p("""Seeds alone do not make training repeatable. Deterministic TensorFlow operations, stateless augmentation seeded by
        step number, a pinned dataset revision, checksums and exact package versions were all required. Two complete runs of
        the classifiers in two separate virtual environments produced identical loss histories, grid-search tables and test
        accuracies, down to the last stored digit.""")


def chapter10(R, M):
    e, F = M["evaluation"], M["fgsm"]["accuracy"]
    R.chapter("Conclusion and Future Scope")
    R.h2("Conclusion")
    R.p(f"""The project delivered a complete deep learning system for leaf disease recognition on modest hardware. A compact CNN
        reached {pct(e['baseline']['accuracy'])} and a fine-tuned MobileNetV2 reached {pct(e['transfer']['accuracy'])} on leaves
        never seen in training. Overfitting and underfitting were demonstrated and treated, hyperparameters were tuned within
        minutes, and predictions can be inspected through activation maps and Grad-CAM. Autoencoders were built for compression,
        denoising and generation and compared with conventional methods.""")
    R.p(f"""The analysis also set limits on what may be claimed. FGSM at ε = 0.01 reduces the transfer model to
        {pct(F['transfer'][3])}, and adversarial training recovers only part of the loss. Eight background pixels predict the
        class with {pct(M['bias_probe']['accuracy'])} accuracy, so results on this dataset should not be taken as field
        performance. A single number would have hidden both facts. The most valuable lesson is that checking a model from
        several directions teaches more than raising its accuracy by one point.""")
    R.h2("Limitations")
    R.bullets(["Ten classes and three crops, with laboratory images on plain backgrounds only.",
               "One training run per model; variation between seeds was not measured.",
               "A test set of 420 images, so small differences are uncertain.",
               "Some images have no leaf identifier and were treated as separate leaves.",
               "FGSM is a single-step attack; stronger iterative attacks were not evaluated.",
               "The Docker image was not built, because Docker was unavailable on the development machine."])
    R.h2("Future Scope")
    R.bullets(["Collect field photographs and test on them, and remove backgrounds by segmentation before training.",
               "Train on all 38 PlantVillage classes at higher resolution on a GPU system such as the DGX Station A100.",
               "Apply PGD adversarial training to the transfer model.",
               "Calibrate confidence and add an option to abstain for unknown crops and unrelated pictures.",
               "Convert the model to TensorFlow Lite with quantization for offline use on phones.",
               "Estimate disease severity in addition to disease type."])


REFS = [
    "S. Savary, L. Willocquet, S. J. Pethybridge, P. Esker, N. McRoberts, and A. Nelson, “The global burden of pathogens and pests on major food crops,” *Nature Ecology & Evolution*, vol. 3, no. 3, pp. 430–439, 2019, doi: 10.1038/s41559-018-0793-y.",
    "S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” *Frontiers in Plant Science*, vol. 7, art. 1419, 2016, doi: 10.3389/fpls.2016.01419.",
    "D. P. Hughes and M. Salathé, “An open access repository of images on plant health to enable the development of mobile disease diagnostics,” arXiv:1511.08060, 2015.",
    "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998, doi: 10.1109/5.726791.",
    "A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” *Communications of the ACM*, vol. 60, no. 6, pp. 84–90, 2017, doi: 10.1145/3065386.",
    "Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, vol. 521, no. 7553, pp. 436–444, 2015, doi: 10.1038/nature14539.",
    "S. Sladojevic, M. Arsenovic, A. Anderla, D. Culibrk, and D. Stefanovic, “Deep neural networks based recognition of plant diseases by leaf image classification,” *Computational Intelligence and Neuroscience*, vol. 2016, art. 3289801, pp. 1–11, 2016, doi: 10.1155/2016/3289801.",
    "K. P. Ferentinos, “Deep learning models for plant disease detection and diagnosis,” *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018, doi: 10.1016/j.compag.2018.01.009.",
    "E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, “A comparative study of fine-tuning deep learning models for plant disease identification,” *Computers and Electronics in Agriculture*, vol. 161, pp. 272–279, 2019, doi: 10.1016/j.compag.2018.03.032.",
    "J. G. A. Barbedo, “Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification,” *Computers and Electronics in Agriculture*, vol. 153, pp. 46–53, 2018, doi: 10.1016/j.compag.2018.08.013.",
    "M. A. Noyan, “Uncovering bias in the PlantVillage dataset,” arXiv:2206.04374, 2022.",
    "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.",
    "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.",
    "S. J. Pan and Q. Yang, “A survey on transfer learning,” *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010, doi: 10.1109/TKDE.2009.191.",
    "C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,” *Journal of Big Data*, vol. 6, art. 60, 2019, doi: 10.1186/s40537-019-0197-0.",
    "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” arXiv:1502.03167, 2015.",
    "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, 2015, arXiv:1412.6980.",
    "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.",
    "G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality of data with neural networks,” *Science*, vol. 313, no. 5786, pp. 504–507, 2006, doi: 10.1126/science.1127647.",
    "P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in *Proc. 25th Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096–1103, doi: 10.1145/1390156.1390294.",
    "D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” arXiv:1312.6114, 2013.",
    "C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” arXiv:1312.6199, 2013.",
    "I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, 2015, arXiv:1412.6572.",
    "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2018, arXiv:1706.06083.",
    "Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004, doi: 10.1109/TIP.2003.819861.",
    "M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” arXiv:1605.08695, 2016.",
    "NVIDIA Corporation, “NVIDIA DGX Station A100 User Guide: Introduction, Getting Started and Hardware Specifications.” [Online]. Available: https://docs.nvidia.com/dgx/dgx-station-a100-user-guide/ (accessed {date}).",
    "NVIDIA Corporation, “NVIDIA A100 Tensor Core GPU datasheet.” [Online]. Available: https://images.nvidia.com/data-center/a100/a100-datasheet.pdf (accessed {date}).",
    "NVIDIA Corporation, “CUDA Toolkit Documentation.” [Online]. Available: https://docs.nvidia.com/cuda/index.html (accessed {date}).",
    "NVIDIA Corporation, “NVIDIA Container Toolkit Overview.” [Online]. Available: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html (accessed {date}).",
    "NVIDIA Corporation, “NVIDIA Multi-Instance GPU User Guide.” [Online]. Available: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ (accessed {date}).",
    "NVIDIA Corporation, “NVIDIA Collective Communication Library (NCCL) Documentation.” [Online]. Available: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html (accessed {date}).",
    "TensorFlow, “Multi-GPU and distributed training.” [Online]. Available: https://www.tensorflow.org/guide/keras/distributed_training (accessed {date}).",
    "TensorFlow, “Mixed precision.” [Online]. Available: https://www.tensorflow.org/guide/mixed_precision (accessed {date}).",
    "S. P. Mohanty, “PlantVillage dataset,” Hugging Face. [Online]. Available: https://huggingface.co/datasets/mohanty/PlantVillage (accessed {date}).",
]


def references(R, M, today):
    R.front("References")
    for i, r in enumerate(REFS, 1):
        p = R.p(f"[{i}]  " + r.replace("{date}", today), align="left", size=10.5, after=2)
        p.paragraph_format.line_spacing = 1.0
        from docx.shared import Cm
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)


def appendix(R, M):
    R.front("Appendix", new_page=False).paragraph_format.space_before = Pt(30)
    R.chap_label = "A"
    R.count = {"Figure": 0, "Table": 0}
    R.d.add_heading("A.1 Repository", 2)
    R.p(f"GitHub: {REPO}", align="left")
    R.d.add_heading("A.2 How to Run", 2)
    R.code("""# 1. environment (Python 3.12)
python -m venv .venv
.venv\\Scripts\\activate                 # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. demo application (uses the trained models in models/)
streamlit run app/app.py

# 3. reproduce everything (under an hour of training on a dual-core CPU)
python -m src.prepare_data
python -m src.run_all

# 4. TensorBoard
tensorboard --logdir logs

# 5. Docker
docker build -t plant-disease .
docker run -p 8501:8501 plant-disease""", size=9)
    R.d.add_heading("A.3 Key Code Listings", 2)
    R.code(extract(ROOT / "src/train_baseline.py", "recalibrate_batchnorm"), title="src/train_baseline.py: batch normalization recalibration")
    R.code(extract(ROOT / "src/gradcam.py", "make_gradcam"), title="src/gradcam.py: Grad-CAM")
    R.code(extract(ROOT / "src/fgsm.py", "fgsm_sign") + "\n\n\n" + extract(ROOT / "src/fgsm.py", "fgsm_attack"), title="src/fgsm.py: FGSM")
    R.code(extract(ROOT / "src/models.py", "build_denoiser"), title="src/models.py: denoising autoencoder")
    R.d.add_heading("A.4 Syllabus-to-Project Mapping", 2)
    R.table(["Syllabus topic", "Project component", "Section"], [
        ["Unit I: tensors, shapes, ranks, computation", "`src/eda.py` tensor demonstration", "3.5, 4.1"],
        ["Unit I: optimizers", "`src/tune.py` optimizer comparison", "8.3"],
        ["Unit I: TensorBoard", "Callbacks in all training scripts, `logs/`", "6.5"],
        ["Unit II: Keras models", "`src/models.py`", "5.3, 5.4, 6.4"],
        ["Unit II: overfit and underfit", "Fit experiment, `train_baseline.py`", "8.1"],
        ["Unit II: hyperparameter tuning", "`src/tune.py` grid search", "8.3"],
        ["Unit II: save and load", "`check_save_load`", "6.4, 8.6"],
        ["Unit III: CNN building blocks", "Baseline CNN", "4.2, 5.3"],
        ["Unit III: transfer learning", "`src/train_transfer.py`", "5.4, 8.4"],
        ["Unit III: augmentation", "`augment` in `src/common.py`", "3.4"],
        ["Unit IV: autoencoder compression", "`src/autoencoder.py`", "5.6, 8.8"],
        ["Unit IV: denoising and VAE", "`src/autoencoder.py`", "5.6, 8.8"],
        ["Unit VI: FGSM and robustness", "`src/fgsm.py`", "5.7, 8.9"],
        ["Unit VI: Docker", "`Dockerfile`", "6.7"],
        ["Unit VI: Streamlit", "`app/app.py`", "6.6, 8.11"],
        ["Practical: activation maps", "`src/gradcam.py`", "8.7"],
        ["Practical: accuracy evaluation", "`src/evaluate.py`", "8.5"],
        ["Practical: pretrained image classification", "MobileNetV2 in script and notebook", "8.4"],
        ["CO2: DGX Station A100 and CUDA", "Theory chapter", "7"]],
        "Syllabus topics and where they are covered", "t-map", [2.6, 2.5, 1.0])
