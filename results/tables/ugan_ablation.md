Results on `ugan_ablation` (mean over images; best per column in bold).

| Method | PSNR (dB) ↑ | PSNR-RGB (dB) ↑ | SSIM ↑ | UIQM ↑ | UCIQE ↑ | ms/img ↓ | N |
|---|---|---|---|---|---|---|---|
| w_ce | **26.82** | 23.52 | **0.880** | 3.039 | **2.043** | 9.0 | 100 |
| wo_gnet | 26.73 | **23.60** | 0.877 | 3.077 | 1.769 | 9.1 | 100 |
| wo_rnet | 26.55 | 23.47 | 0.877 | **3.080** | 1.808 | 8.5 | 100 |
| wo_rnet_gnet | 26.66 | 23.56 | 0.878 | 3.074 | 1.571 | 8.0 | 100 |
| w_rnet_gnet | 26.31 | 21.67 | 0.872 | 2.771 | 1.776 | **2.7** | 100 |
