# Model zoo and historical evaluation records

Nine main-table checkpoints are available in [Release v0.1.1](https://github.com/liujessica114-ctrl/DHQuant/releases/tag/v0.1.1). Their scale and bit-width match the saved tensors, and each has an explicit loading record in a historical evaluation log. Full benchmark verification using the release code remains pending.

The machine-readable [checkpoint manifest](checkpoints.json) records direct download links, SHA-256 hashes, original experiment paths, release filenames, logged settings, and historical results. All nine files total 161,924,214 bytes (154.42 MiB). On 2026-11-18, their GitHub asset sizes and SHA-256 digests were verified against the local manifest; the downloaded [SHA256SUMS](https://github.com/liujessica114-ctrl/DHQuant/releases/download/v0.1.1/SHA256SUMS) also matched all nine entries. Original files remain unchanged.

## Checkpoint inventory

| Scale | W/A bits | Selected update | Logged batch | Logged update budget | Logged seed | Used inverted HCG intervals |
| --- | --- | --- | --- | --- | --- | --- |
| ×4 | 2/2 | 9700 | 16 | 10000 | 2026 | 4 |
| ×4 | 3/3 | 3500 | 16 | 4000 | 3407 | 0 |
| ×4 | 4/4 | 1400 | 16 | 4000 | 3407 | 0 |
| ×3 | 2/2 | 9700 | 16 | 10000 | 2026 | 20 |
| ×3 | 3/3 | 3900 | 16 | 4000 | 2026 | 1 |
| ×3 | 4/4 | 2900 | 16 | 4000 | 2026 | 0 |
| ×2 | 2/2 | 9400 | 16 | 10000 | 2026 | 20 |
| ×2 | 3/3 | 3900 | 16 | 4000 | 2026 | 0 |
| ×2 | 4/4 | 2500 | 16 | 4000 | 2026 | 0 |

Version 0.1.1 loads all nine checkpoints with explicit `quant.bound_policy: legacy`. The four inverted checkpoints remain rejected by the default `ordered` policy. Legacy loading preserves every tensor, with a warning for inverted HCG intervals. All nine passed CPU loading and a fixed synthetic-input comparison against the pre-fix arithmetic (maximum absolute output difference 0). These checks do not verify benchmark metrics. See the [release guide](RELEASE_GUIDE.md) and [paper alignment](PAPER_ALIGNMENT.md).

Logged budgets, completed updates, and selected updates are distinct. In particular, the ×4 W4A4 run is configured for 4,000 updates but its available training log ends at 2,300, and the selected checkpoint is update 1,400. All runs record α=0.5 and G=32. The logged batch size is 16. Nine configurations under `options/historical/` now encode the logged batch, seed and budget using legacy bounds; complete retraining remains unverified.

## Historical test results

Each cell is PSNR / SSIM, copied at the precision recorded in the selected historical test log. These are historical results, not a fresh evaluation of the release code.

| Scale | W/A bits | Set5 | Set14 | B100 | Urban100 | Manga109 |
| --- | --- | --- | --- | --- | --- | --- |
| ×4 | 2/2 | 30.2545 / 0.8562 | 27.3296 / 0.7465 | 26.7366 / 0.7039 | 24.2734 / 0.7136 | 27.0578 / 0.8432 |
| ×4 | 3/3 | 31.3980 / 0.8807 | 28.0893 / 0.7667 | 27.2157 / 0.7221 | 25.2199 / 0.7526 | 29.0108 / 0.8849 |
| ×4 | 4/4 | 32.0393 / 0.8912 | 28.4450 / 0.7771 | 27.4886 / 0.7323 | 25.9062 / 0.7786 | 30.0235 / 0.9024 |
| ×3 | 2/2 | 32.6176 / 0.9066 | 29.2222 / 0.8183 | 28.2715 / 0.7812 | 26.0491 / 0.7952 | 30.2381 / 0.9092 |
| ×3 | 3/3 | 33.7097 / 0.9195 | 29.8852 / 0.8329 | 28.7410 / 0.7944 | 27.1926 / 0.8281 | 32.2085 / 0.9326 |
| ×3 | 4/4 | 34.2309 / 0.9255 | 30.1888 / 0.8397 | 28.9828 / 0.8026 | 27.9033 / 0.8475 | 33.1353 / 0.9419 |
| ×2 | 2/2 | 36.7977 / 0.9544 | 32.5359 / 0.9072 | 31.3829 / 0.8879 | 29.5875 / 0.8973 | 36.1670 / 0.9684 |
| ×2 | 3/3 | 37.6123 / 0.9585 | 33.1642 / 0.9142 | 31.8536 / 0.8950 | 31.0486 / 0.9171 | 37.7975 / 0.9745 |
| ×2 | 4/4 | 37.9349 / 0.9600 | 33.4711 / 0.9174 | 32.0860 / 0.8982 | 31.9448 / 0.9271 | 38.5762 / 0.9768 |

At the manuscript precision (two decimals for PSNR, four for SSIM), 89 of the 90 entries above match Table I. For ×2 W2A2 on Manga109, the log reports 36.1670 dB, which rounds to 36.17, while the manuscript reports 36.16. That discrepancy remains to be resolved.

Some repeated evaluations of the same recorded checkpoint path differ slightly. For ×2 W4A4, the March 2 log matches Table I; the March 3 log reports Urban100 PSNR 31.9458 rather than 31.9448, changing the value rounded to two decimals. The manifest identifies the selected log. Historical logs identify paths rather than file hashes; current hashes pin file identity from release preparation onward.

## Original checkpoint identities

Paths below are relative to the original development checkout. Published release assets use `DHQuant_SwinIRLight_x{scale}_W{bit}A{bit}.pth`; direct download links are in [checkpoints.json](checkpoints.json) and the [README](../README.md#models).

| Scale | W/A bits | Historical checkpoint |
| --- | --- | --- |
| ×4 | 2/2 | `experiments/train_2DQuant_x4_bit2_mse_seed_10000/models/net_Q_9700.pth` |
| ×4 | 3/3 | `experiments/train_2DQuant_x4_bit3_mse_2/models/net_Q_3500.pth` |
| ×4 | 4/4 | `experiments/train_2DQuant_x4_bit4_mse/models/net_Q_1400.pth` |
| ×3 | 2/2 | `experiments/train_2DQuant_x3_bit2_mse_10000iter/models/net_Q_9700.pth` |
| ×3 | 3/3 | `experiments/train_2DQuant_x3_bit3_mse/models/net_Q_3900.pth` |
| ×3 | 4/4 | `experiments/train_2DQuant_x3_bit4_mse/models/net_Q_2900.pth` |
| ×2 | 2/2 | `experiments/train_2DQuant_x2_bit2_mse_10000iter/models/net_Q_9400.pth` |
| ×2 | 3/3 | `experiments/train_2DQuant_x2_bit3_mse/models/net_Q_3900.pth` |
| ×2 | 4/4 | `experiments/train_2DQuant_x2_bit4_mse/models/net_Q_2500.pth` |

SHA-256 checksums and original test-log identifiers for every row are available in [checkpoints.json](checkpoints.json). Upstream full-precision SwinIR initialization weights and datasets are obtained separately as described in the [README](../README.md).
