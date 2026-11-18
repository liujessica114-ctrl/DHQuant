"""Recover only complete, CRC-verified calibration inputs, never missing bytes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

script = Path(__file__).resolve().parents[1] / 'scripts/recover_calibration.py'
spec = importlib.util.spec_from_file_location('calibration_recovery', script)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class CalibrationRecoveryTests(unittest.TestCase):
    def make_archive(self, directory):
        path = Path(directory) / 'complete.pth'
        tensor = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        torch.save({'lq': tensor, 'gt': torch.zeros(2, 3, 16, 16)}, path)
        return path, tensor

    def test_truncated_hr_keeps_exact_lq_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.make_archive(directory)
            tensor, provenance = recovery.recover_lq(path)
            self.assertTrue(torch.equal(tensor, expected))
            blob = path.read_bytes()
            entries = dict(recovery._local_entries(blob))
            lq_end = entries[provenance['storage_record']] + expected.numel() * 4 + 16
            truncated = Path(directory) / 'truncated.pth'
            truncated.write_bytes(blob[:lq_end])
            recovered, record = recovery.recover_lq(truncated)
            self.assertTrue(torch.equal(recovered, expected))
            self.assertTrue(record['storage_crc_verified'])
            self.assertEqual(record['storage_sha256'], provenance['storage_sha256'])

    def test_corrupted_or_truncated_lq_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.make_archive(directory)
            _, provenance = recovery.recover_lq(path)
            blob = path.read_bytes()
            start = dict(recovery._local_entries(blob))[provenance['storage_record']]
            corrupted = bytearray(blob)
            corrupted[start] ^= 1
            bad = Path(directory) / 'corrupt.pth'
            bad.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, 'CRC mismatch'):
                recovery.recover_lq(bad)
            bad.write_bytes(blob[:start + expected.numel() * 4 - 1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                recovery.recover_lq(bad)


if __name__ == '__main__':
    unittest.main()
