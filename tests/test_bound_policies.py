"""Checkpoint compatibility and crossed-bound regression cases."""
import logging
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from basicsr.archs.dhquant_arch import HCGActivationQuantizer, FakeQuantizerAct
from basicsr.models.dhquant_model import DHQuantModel


class BoundPolicyTests(unittest.TestCase):
    def loader(self, policy='ordered'):
        model = DHQuantModel.__new__(DHQuantModel)
        model.opt = {'quant': {'bound_policy': policy}}
        model.logger = logging.getLogger('bound_policy_test')
        return model

    def quantizer(self):
        quantizer = HCGActivationQuantizer(bit=2, n_groups=4, channel_len=4)
        quantizer.labels_buffer = torch.tensor([0, 0, 1, 1])
        quantizer.labels = quantizer.labels_buffer
        quantizer.grouped = quantizer.calibrated = True
        with torch.no_grad():
            quantizer.grouped_lower_bound.copy_(torch.tensor([-1., -.5, 0., 0.]))
            quantizer.grouped_upper_bound.copy_(torch.tensor([1., .5, 0., 0.]))
        return quantizer

    def test_unused_zero_width_groups_survive_roundtrip(self):
        q = self.quantizer()
        x = torch.tensor([[[-2., -.3, .2, 2.]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q.pth'
            torch.save({'params': q.state_dict()}, path)
            restored = HCGActivationQuantizer(bit=2, n_groups=4, channel_len=4)
            self.loader().load_network(restored, path)
            self.assertTrue(torch.equal(q(x), restored(x)))

    def test_legacy_is_explicit_and_preserves_inverted_tensors_and_output(self):
        q = self.quantizer()
        with torch.no_grad():
            q.grouped_lower_bound[1] = .8
            q.grouped_upper_bound[1] = -.4
        x = torch.tensor([[[-2., -.3, .2, 2.]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q.pth'
            torch.save({'params': q.state_dict()}, path)
            restored = HCGActivationQuantizer(bit=2, n_groups=4, channel_len=4)
            with self.assertRaisesRegex(RuntimeError, 'explicit quant.bound_policy=legacy'):
                self.loader().load_network(restored, path)
            with self.assertLogs('bound_policy_test', level='WARNING'):
                self.loader('legacy').load_network(restored, path)
            for key, value in q.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertTrue(torch.equal(q(x), restored(x)))

    def test_legacy_still_rejects_zero_width_nan_and_invalid_labels(self):
        for defect in ('zero', 'nan', 'labels'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                q = self.quantizer()
                with torch.no_grad():
                    if defect == 'zero':
                        q.grouped_upper_bound[0] = q.grouped_lower_bound[0]
                    elif defect == 'nan':
                        q.grouped_upper_bound[0] = float('nan')
                    else:
                        q.labels_buffer[0] = 99
                path = Path(directory) / 'q.pth'
                torch.save({'params': q.state_dict()}, path)
                with self.assertRaises(RuntimeError):
                    self.loader('legacy').load_network(self.quantizer(), path)

    def test_projection_preserves_parameters_valid_intervals_and_unused_groups(self):
        model = self.loader()
        q = self.quantizer()
        with torch.no_grad():
            q.grouped_lower_bound[1] = .8
            q.grouped_upper_bound[1] = -.4
        model.net_Q = nn.Sequential(q)
        before_ids = [id(p) for p in model.net_Q.parameters()]
        model._enforce_ordered_bounds()
        self.assertEqual(before_ids, [id(p) for p in model.net_Q.parameters()])
        self.assertEqual(q.grouped_lower_bound[0], -1.)
        self.assertEqual(q.grouped_upper_bound[0], 1.)
        self.assertGreater(q.grouped_upper_bound[1], q.grouped_lower_bound[1])
        self.assertTrue(torch.equal(q.grouped_upper_bound[2:], torch.zeros(2)))
        self.assertTrue(torch.isfinite(q(torch.zeros(1, 2, 4))).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q.pth'
            torch.save({'params': q.state_dict()}, path)
            self.loader().load_network(self.quantizer(), path)

    def test_projection_handles_tensorwise_bounds_and_legacy_leaves_them_unchanged(self):
        q = FakeQuantizerAct(2)
        q.calibrated = True
        with torch.no_grad():
            q.lower_bound.fill_(2.)
            q.upper_bound.fill_(-1.)
        model = self.loader('legacy')
        model.net_Q = nn.Sequential(q)
        model._enforce_ordered_bounds()
        self.assertEqual(q.lower_bound.item(), 2.)
        self.assertEqual(q.upper_bound.item(), -1.)
        model.opt['quant']['bound_policy'] = 'ordered'
        model._enforce_ordered_bounds()
        self.assertGreater(q.upper_bound.item(), q.lower_bound.item())


if __name__ == '__main__':
    unittest.main()
