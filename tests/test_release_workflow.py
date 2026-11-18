"""CPU checks for DAS, fixed HCG groups, and the public checkpoint workflow."""

from copy import deepcopy
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch import nn
import yaml

from basicsr.archs import build_network
from basicsr.archs.dhquant_arch import HCGActivationQuantizer, QuantLinear
from basicsr.getcalidata import collect_calibration_data
from basicsr.models import build_model
from basicsr.models.dhquant_model import DHQuantModel
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.das import apply_das
from basicsr.utils.options import parse_options


def tiny_network():
    return dict(type='SwinIR', upscale=2, img_size=4, window_size=2, img_range=1.,
                depths=[1, 1], embed_dim=8, num_heads=[2, 2], mlp_ratio=2,
                upsampler='pixelshuffledirect', drop_path_rate=0.)


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_root_logger().setLevel(logging.ERROR)
        torch.set_num_threads(1)

    def test_das_preserves_fp_mapping_and_linear_biases(self):
        set_random_seed(13)
        model = build_network(tiny_network()).eval()
        images = torch.rand(2, 3, 4, 4)
        biases = {name: module.bias.detach().clone()
                  for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and module.bias is not None}
        with torch.no_grad():
            before = model(images)
            pairs = apply_das(model, {'lq': images}, torch.device('cpu'))
            after = model(images)
        self.assertEqual(pairs, 4)
        self.assertTrue(torch.allclose(before, after, rtol=1e-5, atol=1e-6))
        for name, module in model.named_modules():
            if name in biases:
                self.assertTrue(torch.equal(module.bias, biases[name]))

    def test_hcg_caps_groups_by_input_channels_and_keeps_assignments(self):
        layer = QuantLinear({'bit': 2, 'weight_per_channel': True,
                             'hcg': True, 'n_groups': 4})
        layer.set_param(nn.Linear(3, 6))
        quantizer = layer.act_quantizer
        self.assertEqual(quantizer.grouped_lower_bound.numel(), 3)
        x = torch.tensor([[[-3., -.5, 2.], [1., .5, 6.]],
                          [[-2., -.1, 3.], [.5, .2, 5.]]])
        with torch.no_grad():
            quantizer(x)
            labels = quantizer.labels_buffer.clone()
            lower = quantizer.grouped_lower_bound[labels]
            upper = quantizer.grouped_upper_bound[labels]
            scale = (upper - lower) / 3
            expected = ((torch.min(torch.max(x, lower), upper) - lower) / scale).round() * scale + lower
            self.assertTrue(torch.equal(quantizer(x), expected))
            quantizer(x.flip(-1) * 2)
            self.assertTrue(torch.equal(labels, quantizer.labels_buffer))

    def test_calibrate_optimize_and_reload_without_fp_or_calibration_files(self):
        set_random_seed(2026)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp_path, cali_path, quant_path = (root / name for name in ('fp.pth', 'cali.pth', 'quant.pth'))
            images = torch.rand(2, 3, 4, 4)
            torch.save({'params': build_network(tiny_network()).state_dict()}, fp_path)
            torch.save({'lq': images}, cali_path)
            options = dict(
                name='tiny_release_test', model_type='DHQuantModel', is_train=True,
                num_gpu=0, dist=False, bit=2, scale=2, network_Q=tiny_network(),
                quant=dict(das=True, das_alpha=.5, hcg=True, n_groups=2,
                           weight_per_channel=True, qkv_separation=True,
                           hook_per_layer=True, hook_per_block=False),
                cali_data=str(cali_path),
                pathFP=dict(pretrain_network_FP=str(fp_path), strict_load_FP=True),
                path=dict(pretrain_network_Q=None, strict_load_Q=True),
                train=dict(optim_bound=dict(type='Adam', lr=.01),
                           scheduler=dict(type='MultiStepLR', milestones=[10000]),
                           pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean'),
                           feature_loss=dict(type='MSELoss', loss_weight=5., reduction='mean')))
            model = build_model(options)
            frozen = {name: value.detach().clone() for name, value in model.net_Q.named_parameters()
                      if not value.requires_grad}
            groups = {name: value.labels_buffer.clone() for name, value in model.net_Q.named_modules()
                      if isinstance(value, HCGActivationQuantizer)}
            for step in (1, 2):
                model.feed_data({'lq': images})
                model.optimize_parameters(step)
                for name, value in model.net_Q.named_parameters():
                    if value.requires_grad:
                        self.assertIsNotNone(value.grad, name)
                        self.assertTrue(torch.isfinite(value.grad).all(), name)
                    else:
                        self.assertTrue(torch.equal(value, frozen[name]), name)
                for name, module in model.net_Q.named_modules():
                    if name in groups:
                        self.assertTrue(torch.equal(module.labels_buffer, groups[name]))
            with torch.no_grad():
                expected = model.net_Q(images)
            state = deepcopy(model.net_Q.state_dict())
            torch.save({'params': state}, quant_path)
            fp_path.unlink()
            cali_path.unlink()
            options['is_train'] = False
            options['path']['pretrain_network_Q'] = str(quant_path)
            options.pop('pathFP')
            options.pop('cali_data')
            restored = build_model(options)
            self.assertFalse(hasattr(restored, 'net_F'))
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, restored.net_Q(images)))
            self.assertEqual(set(state), set(restored.net_Q.state_dict()))

            group_key = next(key for key in state if key.endswith('labels_buffer'))
            bad = dict(state)
            bad.pop(group_key)
            torch.save({'params': bad}, quant_path)
            with self.assertRaisesRegex(RuntimeError, 'Incomplete quantized checkpoint'):
                build_model(options)
            bad = deepcopy(state)
            bad[group_key].fill_(99)
            torch.save({'params': bad}, quant_path)
            with self.assertRaisesRegex(RuntimeError, 'Invalid fixed HCG'):
                build_model(options)
            torch.save({'params': state}, quant_path)
            options['bit'] = 3
            with self.assertRaisesRegex(RuntimeError, 'bit-width'):
                build_model(options)

    def test_checkpoint_shape_errors_are_not_swallowed(self):
        loader = DHQuantModel.__new__(DHQuantModel)
        loader.logger = get_root_logger()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pth'
            torch.save({'params': nn.Linear(3, 4).state_dict()}, path)
            for strict in (True, False):
                with self.subTest(strict=strict):
                    with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
                        loader.load_network(nn.Linear(3, 2), path, strict=strict)

    def test_actual_train_and_test_entrypoints_reload_new_checkpoint(self):
        """Previously two real optimizer steps saved crossed, unloadable bounds."""
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr = root / 'HR', root / 'LR'
            hr.mkdir()
            lr.mkdir()
            rng = np.random.RandomState(123)
            for index in range(2):
                image = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)
                cv2.imwrite(str(lr / f'{index}x2.png'), image)
                cv2.imwrite(str(hr / f'{index}.png'), image.repeat(2, 0).repeat(2, 1))
            torch.manual_seed(2026)
            fp, cali = root / 'fp.pth', root / 'cali.pth'
            torch.save({'params': build_network(tiny_network()).state_dict()}, fp)
            options = yaml.safe_load((source / 'options/train/train_DHQuant_x2.yml').read_text())
            options.update(name='entrypoint_regression', num_gpu=0, bit=2,
                           cali_data=str(cali), network_Q=tiny_network())
            options['quant']['n_groups'] = 2
            options['pathFP']['pretrain_network_FP'] = str(fp)
            options['datasets']['train'].update(
                dataroot_gt=str(hr), dataroot_lq=str(lr), batch_size_per_gpu=2,
                num_worker_per_gpu=0, gt_size=8, prefetch_mode=None)
            options['datasets']['val_0'].update(dataroot_gt=str(hr), dataroot_lq=str(lr))
            options['train']['total_iter'] = 2
            options['val']['val_freq'] = 1
            options['logger'].update(save_checkpoint_freq=1, print_freq=1)
            collector = dict(manual_seed=0, cali_data_save_path=str(cali),
                             datasets={'train': dict(options['datasets']['train'], phase='train', scale=2)})
            collect_calibration_data(collector)
            train_path = root / 'train.yml'
            train_path.write_text(yaml.safe_dump(options), encoding='utf-8')
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, '-c',
                       'import sys; from basicsr.train import train_pipeline; train_pipeline(sys.argv.pop(1))',
                       str(root), '-opt', str(train_path)]
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout.decode('utf-8', errors='replace')[-5000:])
            checkpoint = root / 'experiments/entrypoint_regression/models/net_Q_2.pth'
            self.assertTrue(checkpoint.is_file())
            fp.unlink()
            cali.unlink()
            options.pop('pathFP')
            options.pop('cali_data')
            options['path']['pretrain_network_Q'] = str(checkpoint)
            options['datasets'] = {'test_1': dict(options['datasets']['val_0'], name='Synthetic')}
            test_path = root / 'test.yml'
            test_path.write_text(yaml.safe_dump(options), encoding='utf-8')
            command = [sys.executable, '-c',
                       'import sys; from basicsr.test import test_pipeline; test_pipeline(sys.argv.pop(1))',
                       str(root), '-opt', str(test_path)]
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            output = result.stdout.decode('utf-8', errors='replace')
            self.assertEqual(result.returncode, 0, output[-5000:])
            self.assertIn('Validation Synthetic', output)

    def test_cli_seed_override_and_repository_relative_paths(self):
        root = Path(__file__).resolve().parents[1]
        options, _ = parse_options(str(root), args=[
            '-opt', str(root / 'options/train/train_DHQuant_x4.yml'),
            '--force_yml', 'manual_seed=123', 'num_gpu=0', 'quant:das=false'])
        observed = torch.rand(4)
        torch.manual_seed(123)
        self.assertTrue(torch.equal(observed, torch.rand(4)))
        self.assertFalse(options['quant']['das'])
        self.assertEqual(options['cali_data'], str(root / 'keydata/cali_data_x4.pth'))
        self.assertTrue(options['datasets']['train']['dataroot_lq'].startswith(str(root)))

    def test_calibration_output_uses_configured_scale_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scale in (2, 4):
                hr, lr = root / f'hr{scale}', root / f'lr{scale}'
                hr.mkdir()
                lr.mkdir()
                for index in range(2):
                    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3) + index
                    cv2.imwrite(str(lr / f'{index}x{scale}.png'), image)
                    cv2.imwrite(str(hr / f'{index}.png'), image.repeat(scale, 0).repeat(scale, 1))
                destination = root / 'keydata' / f'cali_data_x{scale}.pth'
                options = dict(manual_seed=0, cali_data_save_path=str(destination),
                    datasets={'train': dict(name='tiny', type='PairedImageDataset', phase='train',
                        scale=scale, dataroot_gt=str(hr), dataroot_lq=str(lr),
                        filename_tmpl='{}x' + str(scale), io_backend={'type': 'disk'},
                        gt_size=4 * scale, use_hflip=True, use_rot=True,
                        batch_size_per_gpu=2, num_worker_per_gpu=0, prefetch_mode=None)})
                set_random_seed(0)
                collect_calibration_data(deepcopy(options))
                batch = torch.load(destination, map_location='cpu')
                self.assertEqual(tuple(batch['lq'].shape), (2, 3, 4, 4))
                self.assertEqual(tuple(batch['gt'].shape), (2, 3, 4 * scale, 4 * scale))
                before = destination.read_bytes()
                with self.assertRaises(FileExistsError):
                    collect_calibration_data(deepcopy(options))
                self.assertEqual(destination.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
