"""Vendored Sparton — GPU-accelerated SPLADE kernels.

Original source: https://github.com/thongnt99/sparton
License: Apache License 2.0 (see LICENSE file in this directory)

This module is modified from the original sparton_kernel.py:
- Removed module-level print/side effects
- Removed unused imports
- Cleaned up commented-out code

The module is always importable. ``SpartonHead`` is only available
when CUDA and triton are present; callers should catch ``ImportError``
on ``from xpmir.neural._sparton import SpartonHead``.
"""

import torch

_AVAILABLE = False

if torch.cuda.is_available():
    try:
        import triton  # noqa: F401

        _AVAILABLE = True
    except ImportError:
        pass

if _AVAILABLE:
    from xpmir.neural._sparton._kernel import SpartonHead

    __all__ = ["SpartonHead"]
else:
    __all__: list = []
