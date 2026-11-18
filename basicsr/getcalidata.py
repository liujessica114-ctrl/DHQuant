"""Save a reproducible batch of DF2K patches for DAS/HCG initialization."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.utils.options import parse_options


def collect_calibration_data(opt):
    """Save the configured batch and its source paths without constructing a model."""
    dataset_opt = opt['datasets']['train']
    dataset = build_dataset(dataset_opt)
    sampler = EnlargedSampler(dataset, num_replicas=1, rank=0,
                              ratio=dataset_opt.get('dataset_enlarge_ratio', 1))
    sampler.set_epoch(0)
    loader = build_dataloader(dataset, dataset_opt, num_gpu=0, dist=False,
                              sampler=sampler, seed=opt['manual_seed'])
    batch = next(iter(loader), None)
    if batch is None:
        raise ValueError('The calibration dataset contains fewer images than the requested batch size.')
    destination = Path(opt['cali_data_save_path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve an existing calibration set: changing it changes initialization.
    with destination.open('xb') as handle:
        torch.save(batch, handle)
    print(f'Saved {batch["lq"].shape[0]} calibration patches to {destination}')
    return destination


if __name__ == '__main__':
    options, _ = parse_options(str(ROOT), is_train=True)
    collect_calibration_data(options)
