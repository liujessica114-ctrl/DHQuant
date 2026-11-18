# Command reference and migration

DHQuant keeps the original `python basicsr/train.py` / `python basicsr/test.py`
entry points and `-opt ... --force_yml ...` configuration interface. The shell
wrappers are optional. This page maps the development commands to the release
checkout and separates naming changes from changes that affect execution.

## Evaluate a renamed release weight

`path:pretrain_network_Q` can point directly to a renamed `.pth` file at any
existing location. If the ×2 W2A2 asset is stored directly in `experiments/`:

```bash
python basicsr/test.py -opt options/test/test_DHQuant_x2.yml --force_yml bit=2 path:pretrain_network_Q=experiments/DHQuant_SwinIRLight_x2_W2A2.pth quant:bound_policy=legacy
```

Use the ×2 YAML for an ×2 checkpoint. To evaluate ×3, change both the YAML and
the checkpoint to ×3. Renaming the file does not change its architecture or bits.
All nine historical weights use the explicit `legacy` policy.

`name` is optional and controls the results directory only. If omitted, the YAML
default remains in effect, including any `bit4` text in that name even when
`bit=2` selects 2-bit inference. Set `name=test_DHQuant_x2_W2A2` when you want a
matching output label. GPU selection is optional as well; the Python command
uses the first currently visible CUDA device by default.

## What changed

| Development command | Release checkout | Meaning |
| --- | --- | --- |
| `options/train/train_2DQuant_x3.yml` | `options/train/train_DHQuant_x3.yml` | The configuration filename now uses DHQuant; the test YAML follows the same convention. |
| `name=train_2DQuant_x3_bit2_mse_10000iter` | `name=train_DHQuant_x3_bit2_mse_10000iter` | An experiment directory label. Renaming it changes where outputs and resume states are found; it does not rename existing files. |
| `--force_yml gpu=3` | `CUDA_VISIBLE_DEVICES=3` before Python in Bash | The development entry points read `gpu`. The release entry points no longer read it; choose the device before starting Python. `num_gpu=1` is a device count, not GPU ID 1. |
| `--force_yml auto_resume=true` | Standalone `--auto_resume`, before `--force_yml` | In both checked versions, the parser overwrites the YAML value with the command-line flag. Without the flag, `auto_resume=true` alone does not enable automatic resume. |
| `bit=2 train:total_iter=10000` | Same arguments | These still select 2-bit quantization and a 10,000-update budget. |
| `net_Q_9700.pth` | Same checkpoint selection, or its renamed release copy | Update 9,700 can be evaluated even when the training budget is 10,000. |
| Historical unconstrained bounds | Explicit `quant:bound_policy=legacy` | Preserves historical HCG interval behavior. New scale-only configurations default to `ordered`, which can change optimization trajectories. |

The earlier README used `bash scripts/train.sh 4 2` as a compact ×4 example.
That wrapper calls the same Python entry point, but the example scale, experiment
name, checkpoint path, and default bound policy were different from the historical
×3 command. The README now shows direct ×3 Python commands.

## New training run

Prepare the data and FP model described in the [README](../README.md#training).
The example uses the supplied ×3 W4A4 configuration and `ordered` bounds:

```bash
python basicsr/train.py -opt options/train/train_DHQuant_x3.yml --auto_resume --force_yml bit=4
```

The ×3 YAML supplies the name `train_DHQuant_x3_bit4`, a 4,000-update budget,
seed 2026, and batch size 16. Initial calibration uses
`keydata/cali_data_x3.pth`; optimization uses the DF2K loader. The collector is:

```bash
python basicsr/getcalidata.py -opt options/calibration/calibration_x3.yml
```

To test a saved checkpoint from this ordered run:

```bash
python basicsr/test.py -opt options/test/test_DHQuant_x3.yml --force_yml bit=4 path:pretrain_network_Q=experiments/train_DHQuant_x3_bit4/models/net_Q_4000.pth
```

Use the actual saved checkpoint you want to evaluate. No checkpoint is renamed
or selected automatically by the `name` argument.

## Historical training settings

To use the logged ×3 W2A2 settings with legacy bounds:

```bash
CUDA_VISIBLE_DEVICES=3 python basicsr/train.py -opt options/historical/train_DHQuant_x3_W2A2.yml --auto_resume --force_yml bit=2 name=train_DHQuant_x3_bit2_mse_10000iter_legacy train:total_iter=10000
```

This recipe specifies seed 2026, batch size 16, a 10,000-update budget, and
`quant.bound_policy: legacy`. It expects the historical LR calibration input at
`keydata/historical/cali_data_x3_lq.pth`. See the [release guide](RELEASE_GUIDE.md)
for calibration recovery and provenance limitations. Do not replace that input
with newly sampled patches and describe the run as identical to the original.
Full retraining with the release code remains unverified.

The `_legacy` experiment suffix keeps its outputs separate from the new ordered
example. Reload its checkpoints with `quant:bound_policy=legacy`.

## Test an existing historical checkpoint

If the original experiment folder has been copied into this checkout, the user's
original selected checkpoint can be evaluated directly:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_DHQuant_x3.yml --force_yml bit=2 name=test_DHQuant_x3_bit2_mse_10000iter_legacy path:pretrain_network_Q=experiments/train_2DQuant_x3_bit2_mse_10000iter/models/net_Q_9700.pth quant:bound_policy=legacy
```

An absolute checkpoint path is also supported. The old `train_2DQuant_...` path
identifies an existing asset; it is not a requirement to rename the source project.

The release file `DHQuant_SwinIRLight_x3_W2A2.pth` is a byte-identical copy of that
selected checkpoint under a consistent public-facing name. Its hash and original
path are recorded in [checkpoints.json](checkpoints.json). Evaluating it requires
neither FP initialization weights nor calibration files.

## Resume and device notes

- `--auto_resume` searches `experiments/<name>/training_states/` for the latest
  state. It also needs the matching model checkpoint. A model `.pth` alone is
  sufficient for evaluation, but not for restoring optimizer and scheduler state.
- Changing `name` changes the directory searched for resume states. A new
  `train_DHQuant_...` name does not discover the old `train_2DQuant_...` folder.
  Use the original name and the appropriate legacy policy when resuming a migrated
  historical run; cross-version training continuation has not been verified.
- Change GPU 3 to a valid GPU ID on your machine. In PowerShell, run
  `$env:CUDA_VISIBLE_DEVICES = "3"`, then the Python portion of the command.
  For testing on GPU 0, set the value to `"0"`. CPU evaluation uses `num_gpu=0`.
