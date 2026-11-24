# Viva notes

25 likely questions with short answers. Numbers come from `metrics.json`.

**1. What problem does the project solve?**
It classifies a leaf photo into one of ten conditions of tomato, potato and bell pepper, explains the decision, and tests how reliable it is. Best test accuracy: 94.8% on 420 images.

**2. Which dataset did you use and how large is it?**
A subset of PlantVillage: 2,852 RGB images of 256x256, 10 classes, up to 300 per class. Healthy potato has only 152, so I used class weights.

**3. How did you split the data and why?**
By physical leaf, about 70/15/15: 2011/421/420. PlantVillage has several photos per leaf; a random split would put near-copies in train and test and inflate accuracy.

**4. What is a tensor? What are rank and shape?**
A multi-dimensional array with one data type. Rank is the number of axes, shape the size of each. An image batch has rank 4: (32, 128, 128, 3).

**5. Describe your baseline CNN.**
Four blocks of Conv 3x3, BatchNorm, ReLU, MaxPool with 16, 32, 64, 128 filters, then global average pooling, Dense 96, dropout 0.4 and softmax. 111,514 parameters, test accuracy 91.4%.

**6. How do you compute the output size of a convolution?**
floor((n + 2p - k) / s) + 1. For n=128, k=3, p=1, s=1 the output is 128; 2x2 pooling makes it 64.

**7. Why global average pooling instead of Flatten?**
Flattening 8x8x128 into Dense 96 needs 786,528 weights; after GAP it needs 12,384. Fewer parameters means less overfitting.

**8. What is transfer learning and how did you apply it?**
Reusing knowledge from a source task. I froze MobileNetV2 (ImageNet) and trained a new head (12,810 parameters, 89.5% validation), then unfroze from block_13 with learning rate 0.0001 (94.1% validation).

**9. Why MobileNetV2?**
It uses depthwise separable convolutions and inverted residuals with linear bottlenecks, so it is light enough for a CPU or phone.

**10. Why keep BatchNorm frozen during fine-tuning?**
Small batches would give noisy statistics and destroy the ImageNet ones.

**11. How did you demonstrate overfitting and underfitting?**
Three models on 400 images. Underfit: 13.0% train. Overfit: 100.0% train but 65.8% validation. Regularised: gap reduced to 4 points with augmentation, dropout, BatchNorm and early stopping.

**12. What was the hardest bug?**
The baseline trained to 76% but validated at 12%. BatchNorm's moving variance (0.533) had not converged to the real one (0.006) because momentum 0.99 is too slow for 63 steps per epoch. I set momentum 0.9 and re-estimated the statistics on clean training images: validation went from 73.4% to 91.4%.

**13. Is that recalibration data leakage?**
No. It uses only training images and changes no weights, only the running mean and variance.

**14. How did you tune hyperparameters?**
A grid of 12 combinations (learning rate, dropout, batch) on cached MobileNetV2 features. Best: lr 0.001, dropout 0.2, batch 16, chosen by validation loss.

**15. Which optimizers did you compare?**
SGD, SGD with momentum, RMSprop, Adagrad and Adam on the same head. Adaptive ones reduce the loss faster; final accuracies are close.

**16. What augmentation did you use?**
Flips, rotation up to 20 degrees, zoom, shift, brightness and contrast, on training batches only, with stateless seeded ops so runs repeat exactly.

**17. Is transfer learning really better here?**
94.8% against 91.4%; McNemar p = 0.038. Unlikely to be chance alone, but the confidence intervals overlap and the test set is small.

**18. What is Grad-CAM?**
Average the gradient of the class score over each feature map of the last conv layer to get weights, take the weighted sum, apply ReLU and resize. It shows regions that support the prediction.

**19. Explain your autoencoder and its compression ratio.**
64x64x3 = 12,288 bytes become an 8x8x16 code stored as 1,024 bytes: 12:1 at 24.1 dB. JPEG at the same size gets 27.6 dB, so JPEG is still better.

**20. How does the denoiser work and is it useful?**
It sees noisy input (sigma 0.1) and learns the clean target; skip connections keep detail. PSNR: noisy 20.2, Gaussian filter 26.0, denoiser 27.1 dB.

**21. How is a VAE different from an autoencoder?**
The encoder outputs a mean and variance, a code is sampled (reparameterisation trick), and a KL term keeps codes near a standard normal, so random codes can be decoded into new images.

**22. What is FGSM?**
x_adv = clip(x + eps * sign(dL/dx), 0, 1). It needs one gradient. At eps = 0.01 the transfer model drops from 94.8% to 2.6%.

**23. How did you defend against it and what did it cost?**
Adversarial training of the baseline: accuracy at eps 0.01 rises from 14.0% to 46.0%, clean accuracy falls from 91.4% to 86.7%. It is not proven against stronger attacks such as PGD.

**24. Would it work in a real field?**
Not as it is. Images are from a laboratory, and a model that sees only 8 background pixels gets 52.1%, so the dataset is biased. Field data and segmentation are needed.

**25. What is the DGX Station A100 and how would you use it?**
A workstation with four A100 GPUs (up to 320 GB), a 64-core EPYC CPU, 512 GB RAM, NVLink and DGX OS with CUDA, cuDNN and containers. I would use MirroredStrategy for data parallelism with NCCL all-reduce and mixed precision; for this small model one MIG slice would be enough.
