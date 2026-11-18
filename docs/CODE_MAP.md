# Code map

DHQuant is a quantization method applied to SwinIR-light. `SwinIR` remains the backbone class; renaming the inherited network would obscure that distinction and is unnecessary for checkpoint compatibility.

## Manuscript-to-code mapping

| Manuscript | Public code | Configuration |
| --- | --- | --- |
| Section III-B, overall pipeline | `basicsr/models/dhquant_model.py::DHQuantModel` | `model_type: DHQuantModel` |
| Section III-C, Eqs. (3)–(5), DAS | `basicsr/utils/das.py::apply_das` | `quant.das`, `quant.das_alpha` |
| Section III-D, Eqs. (6)–(11), HCG | `basicsr/archs/dhquant_arch.py::HCGActivationQuantizer` | `quant.hcg`, `quant.n_groups` |
| Eq. (2), uniform fake quantization | `FakeQuantizerWeight`, `FakeQuantizerAct`, `HCGActivationQuantizer` | `bit` |
| Section III-E, bound optimization | `DHQuantModel.setup_optimizers`, `optimize_parameters` | `train.optim_bound`, loss settings |
| Section IV-A, SwinIR-light | `basicsr/archs/swinir_arch.py::SwinIR` | `network_Q` |

## Training path

`train.py` parses options, constructs the DF2K/Set5 loaders, and creates `DHQuantModel`. The model loads the FP teacher and student, applies DAS to the student, replaces attention/MLP linear layers with fake-quantized layers, and initializes quantizers with saved calibration patches. HCG labels are fixed after initialization. The optimizer contains only the active upper/lower bounds; network weights and bit-widths are frozen.

The supplied model quantizes q/k/v projections separately, attention projection, fc1/fc2, and intermediate attention tensors. HCG is applied to linear-layer input activations; attention intermediates use the inherited tensor-wise activation quantizer. Convolutions remain FP in the paper configurations.

## Checkpoint path

`test.py` constructs the quantized architecture and loads a complete state dict directly. DAS has already been folded into the saved LayerNorm/Linear parameters, so it is not repeated. HCG labels are restored from buffers and remain fixed. The FP teacher and calibration tensor are only needed for new calibration or teacher-dependent training/evaluation.

Internal parameter keys (including `net_Q` and `labels_buffer`) are preserved. These keys carry checkpoint compatibility; the Python model/module filenames need not retain the old method name.

## Naming migration

| Previous name | Release name |
| --- | --- |
| `TDQuantModel` | `DHQuantModel` |
| `basicsr/models/tdquant_model.py` | `basicsr/models/dhquant_model.py` |
| `basicsr/archs/quant_arch.py` | `basicsr/archs/dhquant_arch.py` |
| `GroupFakeQuantizerAct` | `HCGActivationQuantizer` |
| `smooth_quant_utils.py::apply_smooth_quant` | `das.py::apply_das` |
| `quant.smooth_quant` | `quant.das` |
| `quant.smooth_alpha` | `quant.das_alpha` |
| `quant.act_cag` | `quant.hcg` |

Old YAML files need these key/name updates. Silent aliases are not provided: use the included configurations so the active method is explicit. `DOBI`, `SwinIR`, and the vendored `basicsr` namespace retain their inherited names and attribution.
