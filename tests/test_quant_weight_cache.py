"""Regression checks for quantized weights across calibration and optimization.

Run with the project's tdquant Python environment:
    python -m unittest discover -s tests -p 'test_quant_weight_cache.py' -v
"""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F


# Load the production quantizers without importing unrelated BasicSR networks.
source = Path(__file__).resolve().parents[1] / "basicsr/archs/dhquant_arch.py"
spec = importlib.util.spec_from_file_location("quant_arch_under_test", source)
quant_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant_arch)


def make_layer(kind, *, identity_activation=False, calibrated=True):
    config = {"bit": 2, "weight_per_channel": True, "hcg": False}
    weight = torch.tensor([[1.4, -1.1, 0.2], [-0.5, 0.6, -1.2]])
    inputs = torch.tensor([[1.4, -0.2, 0.5], [-0.6, 1.1, -1.3]])
    if kind == "linear":
        layer = quant_arch.QuantLinear(config)
        layer.set_param(nn.Linear(3, 2, bias=False))
    else:
        layer = quant_arch.QuantConv2d(config)
        layer.set_param(nn.Conv2d(3, 2, 1, bias=False))
        weight = weight.reshape(2, 3, 1, 1)
        inputs = inputs.t().reshape(1, 3, 1, 2)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.weight_quantizer.lower_bound.fill_(-0.75)
        layer.weight_quantizer.upper_bound.fill_(0.75)
        layer.act_quantizer.lower_bound.fill_(-1.0)
        layer.act_quantizer.upper_bound.fill_(1.0)
    layer.weight.requires_grad_(False)
    layer.weight_quantizer.n_bit.requires_grad_(False)
    layer.act_quantizer.n_bit.requires_grad_(False)
    layer.weight_quantizer.calibrated = calibrated
    layer.act_quantizer.calibrated = True
    layer.act_quantizer.identity = identity_activation
    return layer.eval(), inputs


class QuantizedWeightRegressionTests(unittest.TestCase):
    def test_eval_bound_optimization_after_no_grad_forward(self):
        """Evaluation mode must still support successive bound updates."""
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                layer, inputs = make_layer(kind)
                raw_weight = layer.weight.detach().clone()
                quantizers = (layer.weight_quantizer, layer.act_quantizer)
                bounds = [p for q in quantizers for p in (q.lower_bound, q.upper_bound)]
                optimizer = torch.optim.SGD(bounds, lr=0.01)
                with torch.no_grad():
                    layer(inputs)  # Simulate validation before resuming optimization.
                for step in range(2):
                    optimizer.zero_grad()
                    output = layer(inputs)
                    output.square().mean().backward()
                    for q in quantizers:
                        gradients = [p.grad for p in (q.lower_bound, q.upper_bound)]
                        self.assertTrue(all(g is not None for g in gradients))
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                    before = [p.detach().clone() for p in bounds]
                    optimizer.step()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, bounds)))
                    with torch.no_grad():
                        updated_output = layer(inputs)
                    self.assertFalse(torch.allclose(output.detach(), updated_output))
                self.assertTrue(torch.equal(layer.weight, raw_weight))

    def test_first_calibration_pass_does_not_disable_later_weight_quantization(self):
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                layer, inputs = make_layer(kind, identity_activation=True, calibrated=False)
                with torch.no_grad():
                    raw_output = (F.linear(inputs, layer.weight) if kind == "linear"
                                  else F.conv2d(inputs, layer.weight))
                    calibration_output = layer(inputs)
                    quantized_output = layer(inputs)
                    repeated_output = layer(inputs)
                self.assertTrue(layer.weight_quantizer.calibrated)
                self.assertTrue(torch.allclose(calibration_output, raw_output))
                self.assertFalse(torch.allclose(quantized_output, raw_output))
                self.assertTrue(torch.equal(repeated_output, quantized_output))

    def test_no_grad_forward_uses_updated_bounds(self):
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                layer, inputs = make_layer(kind, identity_activation=True)
                with torch.no_grad():
                    before = layer(inputs)
                    layer.weight_quantizer.lower_bound.fill_(-0.25)
                    layer.weight_quantizer.upper_bound.fill_(0.25)
                    after = layer(inputs)
                self.assertFalse(torch.allclose(before, after))

    def test_nested_checkpoint_load_uses_new_bounds(self):
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                old, inputs = make_layer(kind, identity_activation=True)
                new, _ = make_layer(kind, identity_activation=True)
                with torch.no_grad():
                    new.weight_quantizer.lower_bound.fill_(-0.25)
                    new.weight_quantizer.upper_bound.fill_(0.25)
                    parent = nn.Sequential(old)
                    before = parent(inputs)
                    parent.load_state_dict(nn.Sequential(new).state_dict(), strict=True)
                    after = parent(inputs)
                    expected = new(inputs)
                self.assertFalse(torch.allclose(before, expected))
                self.assertTrue(torch.equal(after, expected))

    def test_dtype_change_after_inference(self):
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                layer, inputs = make_layer(kind)
                with torch.no_grad():
                    layer(inputs)
                    layer.double()
                    output = layer(inputs.double())
                self.assertEqual(output.dtype, torch.float64)
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
