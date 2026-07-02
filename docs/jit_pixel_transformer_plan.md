# JiT Pixel Transformer Follow-Up

This is the second step after adding pixel-space `x` prediction to the current
U-Net flow-matching pipeline.

Goal: add a raw-pixel Transformer model inspired by Li and He, "Back to Basics:
Let Denoising Generative Models Denoise" (`liBackBasicsLet2026`), without
introducing a manually chosen latent space.

Suggested sequence:

1. Keep `objective.model_output: x` and `objective.x_prediction.loss_space: clean`.
2. Add a new `model.name: image_jit` beside `image_unet`.
3. Start with MNIST/CIFAR-sized smoke configs before ImageNet:
   - patch size 4 for MNIST/Fashion-MNIST
   - patch size 4 or 8 for CIFAR-10
   - small hidden width/depth before scaling
4. Port only the minimal raw-pixel Transformer pieces first:
   - patchify/unpatchify
   - time conditioning
   - transformer blocks
   - clean-image output head
5. Add JiT-specific features only after the small configs train:
   - large patches for ImageNet
   - qk normalization/RMSNorm/SwiGLU if needed
   - class conditioning and classifier-free guidance
   - paper-specific sampling schedules

Do not replace the U-Net configs until the `x`-prediction U-Net baseline is
measured. The first comparison should isolate prediction target from model
architecture.
