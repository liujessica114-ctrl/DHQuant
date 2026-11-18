"""Small, passive records from the actual HCG initialization forward."""

import hashlib
import json
from functools import partial
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class GroupInitRecorder:
    """Capture descriptors without another forward, clustering, or RNG draw."""

    def __init__(self, network, opt, calibration):
        from basicsr.archs.dhquant_arch import HCGActivationQuantizer

        self.modules = []
        self.records = {}
        self.metadata = {
            'schema_version': 1,
            'stage': 'actual_group_initialization_before_bound_optimization',
            'condition': opt['name'],
            'seed': opt['manual_seed'],
            'network_training': network.training,
            'training_modules': [name for name, module in network.named_modules() if module.training],
            'quant': dict(opt['quant']),
            'calibration_file': opt['cali_data'],
            'calibration_file_sha256': file_sha256(opt['cali_data']),
            'calibration_tensors': {
                key: {'shape': list(calibration[key].shape),
                      'sha256': tensor_sha256(calibration[key])}
                for key in ('lq', 'gt') if key in calibration},
            'kmeans': {'random_state': 0, 'n_init': 10},
        }
        for name, module in network.named_modules():
            if isinstance(module, HCGActivationQuantizer):
                if hasattr(module, '_group_init_observer'):
                    raise RuntimeError('An initialization recorder is already attached')
                module._group_init_observer = partial(self.capture, name)
                self.modules.append(module)

    def capture(self, name, module, shape, minimum, maximum):
        if name in self.records:
            raise RuntimeError('Unexpected repeated grouping initialization: ' + name)
        self.records[name] = {
            'module': name,
            'input_shape': list(shape),
            'channel_axis': module.channel_axis,
            'channel_min': minimum.tolist(),
            'channel_max': maximum.tolist(),
            'labels': module.labels.detach().cpu().tolist(),
            'grouped_lower_bound': module.grouped_lower_bound.detach().cpu().tolist(),
            'grouped_upper_bound': module.grouped_upper_bound.detach().cpu().tolist(),
            'requested_groups': module.n_groups,
        }

    def save(self, path):
        if len(self.records) != len(self.modules):
            raise RuntimeError('Missing actual initialization records; refusing partial export')
        document = dict(self.metadata, module_count=len(self.records),
                        records=list(self.records.values()))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')

    def close(self):
        for module in self.modules:
            del module._group_init_observer
        self.modules.clear()
