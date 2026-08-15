
# GPU-Accelerated Image Super-Resolution

**A CNN that reconstructs sharp 128×128 images from blurry 32×32 inputs — trained twice, once on CPU and once on hand-written CUDA kernels, to measure exactly how much a GPU actually buys you.**

Most super-resolution papers report a GPU speedup number without ever showing the CPU side of the comparison. This project builds both — the same four-layer network, trained under identical conditions, once with PyTorch on CPU and once with CUDA C++ written from scratch — so the 2.31× speedup isn't a claim, it's a measurement.

| | |
|---|---|
| Course | Parallel and Distributed Computing (UCS645) |
| Guidance | Dr. Saif Nalband, Assistant Professor, DCSE |
| Team | Anmolpreet Kaur, Gagandeep |
| Institution | Thapar Institute of Engineering & Technology |

---

## The Problem

Upscaling a low-resolution image with bicubic or bilinear interpolation produces blur, because the fine detail lost during downsampling simply isn't recoverable by a fixed mathematical filter. A trained CNN can do better — it learns to hallucinate plausible edges and texture from patterns seen across thousands of training images. The cost is compute: every training step means hundreds of millions of multiply-accumulate operations across the convolutional layers, repeated every epoch. That's a workload built for a GPU's thousands of parallel cores, not a CPU's handful of sequential ones.

## The Network — FinalSRCNN

A four-layer CNN with a global residual connection:

| Layer | Operation | Kernel | Channels |
|---|---|---|---|
| — | Bicubic upsample (32×32 → 128×128) | — | — |
| 1 | Patch extraction | 9×9 | 1 → 64 |
| 2 | Non-linear mapping | 3×3 | 64 → 64 |
| 3 | Feature compression | 3×3 | 64 → 32 |
| 4 | Reconstruction | 5×5 | 32 → 1 |
| — | Residual add (bicubic output + Layer 4 output) | — | — |

The residual connection means the network only has to learn the *difference* between the bicubic upsample and the true high-resolution image, which converges faster and more stably than learning the mapping outright.

## The Loss Function

Pixel accuracy alone tends to produce soft, over-smoothed outputs, so training uses a combined loss:

```
L = MSE(pred, target) + 0.05 × MSE(φ(pred), φ(target))
```

where `φ` extracts features from the first eight layers of a pre-trained VGG16. The MSE term keeps the output numerically accurate; the VGG perceptual term pushes it toward outputs that look sharp to a human eye rather than merely scoring well pixel-by-pixel.

## Built Twice, On Purpose

**PyTorch (CPU baseline)** — the reference implementation, used to validate correctness and establish the baseline training time.

**CUDA C++ (GPU, from scratch)** — every stage of the training loop hand-written as a CUDA kernel, not delegated to cuDNN:

| Kernel | Job |
|---|---|
| `k_upsample4` | 4× bilinear upsampling |
| `k_conv2d_fwd` | Forward convolution |
| `k_conv2d_bwd_w` / `bwd_b` / `bwd_x` | Backward gradients — weights, bias, input |
| `k_relu_bwd` | ReLU backward pass |
| `k_mse_loss` | MSE loss and gradient |
| `k_adam` | Adam optimizer parameter update |

Writing these by hand — rather than calling a framework — was the point: it forces an explicit mapping between each neural network operation and the GPU thread/block/memory model, since every output pixel of a convolution is independent and can be computed by its own CUDA thread.

## Dataset

300 grayscale images self-curated from the Picsum Photos API, covering landscapes, urban scenes, portraits, textures, and abstract patterns.

- HR target: 128×128 · LR input: 32×32 (bicubic downscale)
- Pixel values normalized to [0, 1]
- Fetched with deterministic seeds (`seed = i × 13` for i = 1 to 300) so the dataset is fully reproducible
- Identical data pipeline used for both the CPU and GPU runs — no augmentation, to keep the timing comparison fair

## Results

Same architecture, same 60 epochs, same batch size of 8, same Adam learning rate (0.001) — the only variable is CPU vs. GPU execution.

| Metric | CPU | GPU (CUDA) | Difference |
|---|---|---|---|
| Total training time | 1,447.88 s | 626.26 s | **2.31× faster — 56.7% less time** |
| Final MSE loss (epoch 60) | ~0.110 | ~0.104 | 5.5% lower |
| Epochs / batch size | 60 / 8 | 60 / 8 | — |

The loss curve drops from ~0.137 to ~0.104 over 60 epochs with no divergence, and the GPU run converges to a slightly lower final loss. The 2.31× speedup, while substantial, falls short of the theoretical GPU-to-CPU throughput ratio — attributable to the relatively small batch size, moderate feature map dimensions, and CUDA memory transfer overhead at this model scale.

Visually, the reconstructed 128×128 output shows continuous edges and restored texture where the 32×32 input shows visible pixelation and blocky boundaries — evidence that the perceptual loss term is doing real work, not just optimizing a number.

## Where This Fits

Most published CUDA benchmarking for super-resolution targets large models like EDSR or ESRGAN, using high-level frameworks that hand GPU execution off to cuDNN. This project deliberately goes the other way — a small, deployment-sized network, with every kernel written by hand — to get a direct, unabstracted measurement of GPU speedup at a scale that's understudied in the literature.

## References

Built on foundational work in SR architectures (SRCNN, EDSR, ESRGAN, RCAN), residual learning (He et al., ResNet), and CUDA programming (Kirk & Hwu, *Programming Massively Parallel Processors*). Full citations in the project report.
