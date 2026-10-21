# Reproduction notes

## What reproduces

- **The colour prior (Fig. 1).** Red is attenuated and green over-strengthened, on synthetic UGAN data (R +45, G −22) and on real UIEB images (R +22, G −9).
- **The synthetic benchmark.** On UGAN test-100 the method is the best of all compared methods on PSNR-RGB (23.63 dB), PSNR-Y (26.38 dB), SSIM (0.884) and UCIQE. SSIM is within 0.007 of the paper's 0.891. PSNR is about 2 dB lower, after 30.7 of the paper's 50 epochs.
- **Fig. 11.** After enhancement the UGAN channel deviations shrink to R −4.6, G −7.8, B −2.2.
- **The loss weights.** λ1 = 1, λ2 = 0.1 is among the best settings.

## What does not reproduce

!!! warning "Generalisation to real images"
    Trained only on synthetic UGAN pairs (as in the paper), the model does **not** beat the raw input
    on UIEB test-90 PSNR / SSIM (17.57 dB / 0.818 vs 17.79 dB / 0.865). The channel analysis shows
    why: the synthetic red loss (+45) is about twice the real one (+22), so the network
    over-compensates red on real photos (deviation −24.3 after enhancement). It still has the highest
    UCIQE on UIEB and Challenging-60, and the second-highest UIQM on Challenging-60.

!!! warning "Ablation"
    At an equal 4,000 iterations, the contrast module (Ce) is worth **+1.85 dB** PSNR-RGB
    (23.52 vs 21.67 dB). Removing Rnet, Gnet or both changes PSNR by **less than 0.1 dB**, so the
    roughly 3 dB contribution of the colour modules in the paper's Table 4 is not reproduced. The
    multi-scale convolutions in Ce can learn the colour correction themselves.

- **NIQE** is worse than the raw input for the learned model on every set.
- The learned confidence maps \(W_{CE}\) and \(W_{GC}\) are almost uniform, so most of the adaptation happens in the \(Z\) refinement branches.

## Findings about the paper's setup

**PSNR is computed on RGB.** The paper says its PSNR uses the luminance channel. Its *input* PSNR on
UGAN (16.85 dB) matches our RGB PSNR of the raw input (16.77 dB), not the luminance PSNR (25.39 dB).
These docs therefore compare against the paper on PSNR-RGB and report both.

**Only λ2/λ1 matters.** Adam's update does not change when the whole loss is scaled, so pairs with the
same ratio behave the same: (1, 0.1) and (10, 1) both give 23.42 dB, and (0.1, 0.1), (1, 1) and
(10, 10) give 23.24–23.33 dB. Only λ2/λ1 = 100 clearly hurts (21.90 dB).

## Engineering notes

**fp16 mixed precision broke training.** With fp16 autocast, runs on A10G and L40S plateaued at
L1 ≈ 0.15–0.35, worse than the identity mapping, while fp32 runs reached L1 ≈ 0.07 within 250
iterations. Training is now fp32/TF32 by default (`--amp` opts in), and CLAHE always runs in fp32.

**Placement of (1 − Ir).** Applying \((1-I_r)\) *before* the biased 1×1 and tail convolutions does not
gate the term; the smoke test caught this. It now multiplies the whole compensation term, as Eq. (1)
states.

**Deviations from the paper.**

- Training length: 30.7 epochs instead of 50, because of the time budget.
- UIEB evaluation: the official seed-42 test-90 split is used for the real-with-reference results.
- All images are resized to 256×256.
