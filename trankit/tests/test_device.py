# Modified by Guillaume Loignon, 2026: device selection (cpu/cuda/mps) and related changes.
# Derived from Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.

import unittest
from types import SimpleNamespace

from trankit.device import resolve_device, transformer_attention_mask_dtype


class FakeTorch:
    float16 = object()
    float32 = object()

    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )

    @staticmethod
    def device(name):
        return SimpleNamespace(type=name)


class DeviceTests(unittest.TestCase):
    def test_legacy_gpu_flag_selects_cuda_when_available(self):
        device = resolve_device(torch_module=FakeTorch(cuda_available=True))
        self.assertEqual(device.type, 'cuda')

    def test_legacy_gpu_flag_falls_back_to_cpu(self):
        device = resolve_device(torch_module=FakeTorch(cuda_available=False))
        self.assertEqual(device.type, 'cpu')

    def test_explicit_mps_requires_available_backend(self):
        torch_module = FakeTorch(mps_available=True)
        self.assertEqual(resolve_device('mps', torch_module=torch_module).type, 'mps')
        with self.assertRaises(RuntimeError):
            resolve_device('mps', torch_module=FakeTorch())

    def test_rejects_unknown_device(self):
        with self.assertRaises(ValueError):
            resolve_device('tpu', torch_module=FakeTorch())

    def test_attention_mask_dtype_is_device_appropriate(self):
        torch_module = FakeTorch()
        self.assertIs(transformer_attention_mask_dtype('mps', torch_module), torch_module.float32)
        self.assertIs(transformer_attention_mask_dtype('cpu', torch_module), torch_module.float16)


if __name__ == '__main__':
    unittest.main()