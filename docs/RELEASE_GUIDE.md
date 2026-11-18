# Release and reproduction guide (0.1.1)

The release supports two explicit quantization-bound policies. They share the
existing feature/output loss definition; no loss equation was silently replaced.

| Policy | Loading | Optimization | Intended use |
|---|---|---|---|
| `ordered` (default) | Rejects inverted/zero-width active intervals | Projects crossed/zero-width active intervals to a positive interval after initialization and each update | New runs with reloadable checkpoints |
| `legacy` | Allows finite, nonzero inverted **HCG** intervals unchanged, with a warning | Retains the historical unconstrained updates | Historical paper checkpoints and historical recipes |

Both policies reject missing quantization state, non-finite bounds, invalid labels,
bit-width mismatches and incompatible shapes. Unused groups may retain `[0,0]`
because no channel indexes them. Legacy loading never swaps or edits weights.

Newly saved files include `dhquant_meta` recording the policy and
`legacy_cross_sample_v1` loss definition. This metadata does not override the
user's explicit configuration. Use `legacy` when reloading a legacy run.

## Evaluate a historical checkpoint

After obtaining the model asset and benchmark data, run from the repository root:

```bash
python basicsr/test.py -opt options/test/test_DHQuant_x4.yml --force_yml bit=2 name=test_DHQuant_x4_bit2_legacy quant:bound_policy=legacy path:pretrain_network_Q=release_assets/checkpoints/DHQuant_SwinIRLight_x4_W2A2.pth
```

The equivalent shell wrapper is:

```bash
bash scripts/test.sh 4 2 release_assets/checkpoints/DHQuant_SwinIRLight_x4_W2A2.pth quant:bound_policy=legacy
```

Use the matching scale/bit asset and configuration; add `num_gpu=0` for CPU.
These commands require neither FP initialization weights nor calibration files.
`release_assets/` is the local staging location used by the maintainer and is
excluded from source control. Users may supply a different checkpoint path.
Public download URLs are available in [checkpoints.json](checkpoints.json) and
[Release v0.1.1](https://github.com/liujessica114-ctrl/DHQuant/releases/tag/v0.1.1).

## Historical training settings

`options/historical/train_DHQuant_x{2,3,4}_W{2,3,4}A{2,3,4}.yml` contains nine
explicit scale/bit recipes. They specify the logged batch size 16, seed, and
10,000-update (2-bit) or 4,000-update (3/4-bit) budget. For example:

```bash
python basicsr/train.py -opt options/historical/train_DHQuant_x4_W2A2.yml
```

These recipes expect `keydata/historical/cali_data_x4_lq.pth` (or the corresponding
scale), DF2K images, the official FP SwinIR-light weight, and validation images.
The selected historical checkpoint may precede the final update. The release
source also contains earlier cache/freezing fixes and is not an immutable copy
of the source used for every historical experiment. Matching settings alone
does not establish exact retraining reproducibility.

The regular `scripts/train.sh SCALE BIT` command uses the same logged batch,
budget and seed choices, but defaults to the **ordered** policy. It is a repaired
training workflow and must be evaluated independently before reporting results.

## Recover calibration inputs from a truncated legacy archive

The migrated ×3 and ×4 calibration files were truncated in their HR storage.
Their original 32×3×64×64 LR storage records and metadata remain complete and
pass the ZIP size/CRC checks. Calibration and DAS consume `lq`; they do not need
the saved HR patches. Bound optimization still needs the normal DF2K loader.

For a trusted local legacy archive:

```bash
python scripts/recover_calibration.py /path/to/old/cali_data_x4.pth keydata/historical/cali_data_x4_lq.pth
```

The tool accepts the supported uncompressed, little-endian float32 legacy format,
reads only whitelisted tensor metadata, verifies each required record's CRC and
size, and writes only `{'lq': tensor}`. It refuses missing or corrupted LR bytes,
unsupported tensor layouts and existing destinations. The source is never modified.
A companion JSON records source/output hashes and the verified LR storage hash.

This recovers the exact surviving input tensor, **not** the missing HR data or
the full original file. No crops are regenerated. Integrity of the surviving
record does not establish that the file was the calibration asset used in each
historical experiment; that remains tied to the available historical evidence.

## Validation and remaining publication work

The 0.1.1 validation uses Python 3.8.20, PyTorch 2.4.1+CPU and TorchVision 0.19.1+CPU
on Windows. The test suite covers actual two-step training, saving, reloading
without FP/calibration files, and evaluation. All nine historical checkpoints
load in explicit legacy mode and preserve their tensors. On one fixed synthetic
input per model, output is bitwise equal to the pre-fix arithmetic with its
rejecting load guard bypassed **in audit code only**.

These checks do not replace GPU training or Set5/Set14/B100/Urban100/Manga109
benchmark evaluation. Before describing a final paper reproduction release:

1. Evaluate all nine assets on the five benchmark datasets in the intended
   environment, recording versions and comparing the resulting metrics.
2. Resolve the remaining manuscript/loss-definition and metric discrepancies
   documented in `PAPER_ALIGNMENT.md`; preserve an explicit version for changes.
3. Complete paper citation and ablation correspondence. Keep local calibration
   assets separate from the source distribution.

Weight publication is complete: all nine assets and `SHA256SUMS` are available
in [Release v0.1.1](https://github.com/liujessica114-ctrl/DHQuant/releases/tag/v0.1.1),
with direct links in `checkpoints.json`. On 2026-11-18, GitHub asset sizes and
SHA-256 digests matched the local manifest, and the downloaded checksum file
matched all nine entries. This verifies upload integrity, not benchmark results.

Build source distributions with `python setup.py sdist`. The manifest includes
code, configurations, scripts, tests and documentation, and excludes `.review`.
Do not archive the entire development folder: `2D_smooth_AHCPTQ/` contains an
unrelated embedded repository and old logs. Historical calibration files and
model assets are kept outside the source distribution.
