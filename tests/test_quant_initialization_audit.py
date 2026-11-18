"""Regression checks for passive grouping export and active-bound freezing."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from basicsr.archs.dhquant_arch import HCGActivationQuantizer, QuantLinear
from basicsr.archs.swinir_arch import BasicLayer
from basicsr.models.dhquant_model import DHQuantModel
from basicsr.utils.quant_audit import GroupInitRecorder


def tiny_dobi(x, **kwargs):
    # Only the observer lifecycle is under test here. Real GPU runs use DOBI.
    return float(x.min()), float(x.max())


class InitializationAuditTests(unittest.TestCase):
    def test_checkpointing_preserves_outputs_and_active_bound_gradients(self):
        torch.manual_seed(2026)
        model = DHQuantModel.__new__(DHQuantModel)
        model.net_Q = BasicLayer(dim=4, input_resolution=(2, 2), depth=2,
                                 num_heads=1, window_size=2, mlp_ratio=2,
                                 use_checkpoint=False).eval()
        model.net_F = deepcopy(model.net_Q)
        model.opt = {'bit': 2, 'quant': {'qkv_separation': True, 'weight_per_channel': True},
                     'train': {'optim_bound': {'type': 'Adam', 'lr': .01}}}
        model.optimizers = []
        model.build_quantized_network()
        inputs = torch.randn(2, 4, 4)
        with patch('basicsr.archs.dhquant_arch.DOBI', side_effect=tiny_dobi):
            with torch.no_grad():
                model.net_Q(inputs, (2, 2))
        model.setup_optimizers()
        recomputed = deepcopy(model.net_Q)
        recomputed.use_checkpoint = True
        # Two backwards exercise both the frozen-input first block and the
        # differentiable later block, with no cache/reinitialization shortcuts.
        for _ in range(2):
            model.net_Q.zero_grad()
            recomputed.zero_grad()
            regular = model.net_Q(inputs, (2, 2))
            checked = recomputed(inputs, (2, 2))
            self.assertTrue(torch.equal(regular, checked))
            regular.square().mean().backward()
            checked.square().mean().backward()
            for (name, a), (_, b) in zip(model.net_Q.named_parameters(), recomputed.named_parameters()):
                if a.requires_grad:
                    self.assertIsNotNone(a.grad, name)
                    self.assertIsNotNone(b.grad, name)
                    self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-8, rtol=1e-5), name)
                else:
                    self.assertIsNone(b.grad, name)

    def test_calibration_disables_random_layers_for_every_condition(self):
        model = DHQuantModel.__new__(DHQuantModel)
        model.net_Q = nn.Sequential(nn.Dropout(p=1.0))
        model.net_Q.train()
        model.device = torch.device('cpu')
        model.opt = {'quant': {}}
        model.cali_data = {'lq': torch.ones(2, 3)}
        seen = []
        model.net_Q.register_forward_hook(lambda module, args, output: seen.append(output.clone()))
        model.calibration()
        self.assertFalse(model.net_Q.training)
        self.assertTrue(torch.equal(seen[0], torch.ones(2, 3)))

    def test_export_is_passive_and_only_records_true_initialization(self):
        x = torch.tensor([[[1., -2., 4., 8.], [3., 0., 6., 10.]],
                          [[2., -1., 5., 9.], [-1., 1., 2., 12.]]])
        with tempfile.TemporaryDirectory() as directory:
            cali_path = Path(directory) / 'calibration.pth'
            torch.save({'lq': x}, cali_path)
            opt = {'name': 'unit_test', 'manual_seed': 2026,
                   'quant': {}, 'cali_data': str(cali_path)}
            torch.manual_seed(2026)
            plain = HCGActivationQuantizer(bit=2, n_groups=2)
            observed = HCGActivationQuantizer(bit=2, n_groups=2)
            observed.load_state_dict(plain.state_dict())
            network = nn.Sequential(observed)
            recorder = GroupInitRecorder(network, opt, {'lq': x})
            before = torch.get_rng_state()
            numpy_before = np.random.get_state()
            with patch('basicsr.archs.dhquant_arch.DOBI', side_effect=tiny_dobi):
                with torch.no_grad():
                    plain(x)
                    network(x)
                    network(x)  # Must not emit a second initialization record.
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(plain.labels, observed.labels))
            self.assertTrue(torch.equal(plain.grouped_lower_bound, observed.grouped_lower_bound))
            path = Path(directory) / 'stats.json'
            recorder.save(path)
            record = json.loads(path.read_text())['records'][0]
            self.assertEqual(record['channel_min'], [-1., -2., 2., 8.])
            self.assertEqual(record['channel_max'], [3., 1., 6., 12.])
            self.assertEqual(record['labels'], observed.labels.tolist())
            recorder.close()
            self.assertFalse(hasattr(observed, '_group_init_observer'))

            recovered = HCGActivationQuantizer(bit=2, n_groups=2, channel_len=4)
            recovered.loaded_from_checkpoint = True
            recovery_recorder = GroupInitRecorder(nn.Sequential(recovered), opt, {'lq': x})
            with torch.no_grad():
                recovered(x)
            self.assertEqual(recovery_recorder.records, {})
            with self.assertRaises(RuntimeError):
                recovery_recorder.save(path)
            recovery_recorder.close()

    def test_optimizer_excludes_backbone_bits_and_unused_group_bounds(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                model = DHQuantModel.__new__(DHQuantModel)
                model.net_F = nn.Linear(4, 4)
                linear = QuantLinear({'bit': 2, 'weight_per_channel': True,
                                      'hcg': grouped, 'n_groups': 2})
                linear.set_param(nn.Linear(4, 4))
                if grouped:
                    with patch('basicsr.archs.dhquant_arch.DOBI', side_effect=tiny_dobi):
                        with torch.no_grad():
                            linear.act_quantizer(torch.randn(2, 3, 4))
                model.net_Q = nn.Sequential(linear, nn.LayerNorm(4))
                model.opt = {'train': {'optim_bound': {'type': 'Adam', 'lr': .01}}}
                model.optimizers = []
                model.setup_optimizers()
                self.assertFalse(any(p.requires_grad for p in model.net_F.parameters()))
                self.assertFalse(linear.weight.requires_grad)
                self.assertFalse(linear.bias.requires_grad)
                self.assertFalse(model.net_Q[1].weight.requires_grad)
                self.assertFalse(linear.weight_quantizer.n_bit.requires_grad)
                self.assertFalse(linear.act_quantizer.n_bit.requires_grad)
                self.assertEqual(len(model.optim_bound_param_names), 4)
                self.assertEqual(sum(p.requires_grad for p in model.net_Q.parameters()), 4)
                if grouped:
                    self.assertFalse(linear.act_quantizer.lower_bound.requires_grad)
                    self.assertFalse(linear.act_quantizer.upper_bound.requires_grad)
                    self.assertTrue(linear.act_quantizer.grouped_lower_bound.requires_grad)
                    self.assertTrue(linear.act_quantizer.grouped_upper_bound.requires_grad)


if __name__ == '__main__':
    unittest.main()
