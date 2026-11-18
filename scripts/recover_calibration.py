"""Recover a CRC-verified LR tensor from legacy, possibly truncated calibration ZIPs.

Only contiguous float32 tensor metadata from the legacy torch.save format is
accepted. Missing HR bytes are never fabricated. Originals are read-only.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import struct
import zlib

import numpy as np
import torch


class _Storage:
    def __init__(self, key, count):
        self.key, self.count = str(key), count


def _tensor_metadata(storage, offset, shape, stride, *unused):
    return dict(storage=storage, offset=offset, shape=tuple(shape), stride=tuple(stride))


class _MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('torch', 'FloatStorage'): 'float32',
            ('torch._utils', '_rebuild_tensor'): _tensor_metadata,
            ('torch._utils', '_rebuild_tensor_v2'): _tensor_metadata,
            ('collections', 'OrderedDict'): OrderedDict,
        }
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported calibration metadata: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, record):
        if (len(record) != 5 or record[0] != 'storage' or record[1] != 'float32'
                or not isinstance(record[4], int) or record[4] <= 0):
            raise ValueError('Only legacy float32 storage metadata is supported.')
        return _Storage(record[2], record[4])


def _local_entries(blob):
    offset = 0
    while True:
        offset = blob.find(b'PK\x03\x04', offset)
        if offset < 0 or offset + 30 > len(blob):
            return
        fields = struct.unpack_from('<IHHHHHIIIHH', blob, offset)
        name_size, extra_size = fields[-2:]
        start = offset + 30 + name_size + extra_size
        name = blob[offset + 30:offset + 30 + name_size]
        if fields[3] == 0 and fields[2] & 8 and start <= len(blob):
            try:
                yield name.decode('utf-8'), start
            except UnicodeDecodeError:
                pass
        offset += 4


def _verified_record(blob, start, size):
    end = start + size
    if end + 16 > len(blob) or blob[end:end + 4] != b'PK\x07\x08':
        raise ValueError('Requested record is truncated or lacks its ZIP data descriptor.')
    crc, compressed, uncompressed = struct.unpack_from('<III', blob, end + 4)
    data = blob[start:end]
    if compressed != size or uncompressed != size or zlib.crc32(data) != crc:
        raise ValueError('ZIP record size or CRC mismatch; refusing recovery.')
    return data


def recover_lq(source):
    """Return an exact LR tensor and provenance; do not write to the source."""
    source = Path(source)
    blob = source.read_bytes()
    entries = dict(_local_entries(blob))
    metadata_names = [name for name in entries if name.endswith('/data.pkl')]
    if len(metadata_names) != 1:
        raise ValueError('Expected exactly one uncompressed legacy data.pkl record.')
    name = metadata_names[0]
    byteorder_name = name[:-len('data.pkl')] + 'byteorder'
    if byteorder_name in entries:
        if _verified_record(blob, entries[byteorder_name], 6) != b'little':
            raise ValueError('Only little-endian calibration archives are supported.')
    start = entries[name]
    stop = next((position + 1 for opcode, _, position in pickletools.genops(blob[start:])
                 if opcode.name == 'STOP'), None)
    if stop is None:
        raise ValueError('Incomplete tensor metadata.')
    metadata = _MetadataReader(io.BytesIO(_verified_record(blob, start, stop))).load()
    item = metadata['lq']
    shape, stride, storage = item['shape'], item['stride'], item['storage']
    expected_stride, count = [], 1
    for dimension in reversed(shape):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError('Invalid tensor dimensions.')
        expected_stride.insert(0, count)
        count *= dimension
    if (len(shape) != 4 or shape[1] != 3 or item['offset'] != 0
            or stride != tuple(expected_stride) or storage.count != count):
        raise ValueError('Expected a contiguous, zero-offset NCHW RGB calibration tensor.')
    storage_name = name[:-len('data.pkl')] + 'data/' + storage.key
    if storage_name not in entries:
        raise ValueError('The LR storage is missing from the source.')
    data = _verified_record(blob, entries[storage_name], count * 4)
    tensor = torch.from_numpy(np.frombuffer(data, dtype='<f4').copy().reshape(shape))
    if not torch.isfinite(tensor).all():
        raise ValueError('Recovered calibration tensor contains non-finite values.')
    provenance = {
        'source_file': str(source.resolve()),
        'source_sha256': hashlib.sha256(blob).hexdigest(),
        'storage_record': storage_name,
        'storage_crc_verified': True,
        'storage_sha256': hashlib.sha256(data).hexdigest(),
        'shape': list(shape),
        'dtype': 'float32',
        'byte_order': 'little',
        'scope': 'Original complete LR storage only; HR was not recovered or synthesized.',
    }
    return tensor, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Source and output must differ.')
    provenance_path = args.output.with_suffix(args.output.suffix + '.json')
    if args.output.exists() or provenance_path.exists():
        parser.error('Output or provenance already exists; choose a new output path.')
    tensor, provenance = recover_lq(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save({'lq': tensor}, handle)
    provenance['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    with provenance_path.open('x', encoding='utf-8') as handle:
        json.dump(provenance, handle, indent=2)
    print(f'Saved {tensor.shape[0]} original LR patches to {args.output}')


if __name__ == '__main__':
    main()
