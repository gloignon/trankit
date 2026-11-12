# Modified by Guillaume Loignon, 2026: device selection (cpu/cuda/mps) and related changes.
# Derived from Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.

"""Device selection helpers shared by the inference and training pipelines."""

import torch


SUPPORTED_DEVICES = ('cpu', 'cuda', 'mps')


def resolve_device(device=None, gpu=True, torch_module=None):
    """Resolve an explicit device or preserve Trankit's legacy ``gpu`` behavior."""
    torch_module = torch if torch_module is None else torch_module

    if device is None:
        if gpu and torch_module.cuda.is_available():
            return torch_module.device('cuda')
        return torch_module.device('cpu')

    requested = device.type if hasattr(device, 'type') else str(device).lower()
    if requested not in SUPPORTED_DEVICES:
        choices = ', '.join(SUPPORTED_DEVICES)
        raise ValueError("Unsupported device '{}'. Choose one of: {}.".format(device, choices))

    if requested == 'cuda':
        available = torch_module.cuda.is_available()
    elif requested == 'mps':
        mps_backend = getattr(getattr(torch_module, 'backends', None), 'mps', None)
        available = mps_backend is not None and mps_backend.is_available()
    else:
        available = True

    if not available:
        raise RuntimeError(
            "Requested device '{}' is unavailable in this PyTorch build/runtime.".format(requested)
        )
    return torch_module.device(requested)


def transformer_attention_mask_dtype(device, torch_module=None):
    """Keep MPS masks in float32 and preserve legacy precision elsewhere."""
    torch_module = torch if torch_module is None else torch_module
    device_type = device.type if hasattr(device, 'type') else str(device).split(':', 1)[0].lower()
    return torch_module.float32 if device_type == 'mps' else torch_module.float16

class DeviceSafeCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """CrossEntropyLoss that is evaluated on CPU for MPS tensors.

    torch 2.0.x's MPS cross_entropy/nll_loss kernels crash (SIGBUS/segfault) on current macOS.
    The logits are small, and ``.cpu()`` is differentiable, so gradients still reach the MPS graph.
    """

    def forward(self, input, target):
        if input.device.type == 'mps':
            input, target = input.cpu(), target.cpu()
        return super().forward(input, target)
