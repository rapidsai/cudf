# Copyright (c) 2025, NVIDIA CORPORATION.

import contextvars
from contextlib import contextmanager

from numba import config as numba_config

_current_nrt_context: contextvars.ContextVar = contextvars.ContextVar(
    "current_nrt_context"
)


class CaptureNRTUsage:
    """
    Context manager for determining if NRT is needed.
    Managed types may set use_nrt to be true during
    instantiation to signal that NRT must be enabled
    during code generation.
    """

    def __init__(self):
        self.use_nrt = False

    def __enter__(self):
        self._token = _current_nrt_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_nrt_context.reset(self._token)


@contextmanager
def nrt_enabled():
    """
    Context manager for enabling NRT via the numba
    config. CUDA_ENABLE_NRT may be toggled dynamically
    for a single kernel launch, so we use this context
    to enable it for those that we know need it.
    """
    original_value = numba_config.CUDA_ENABLE_NRT
    numba_config.CUDA_ENABLE_NRT = True
    try:
        yield
    finally:
        numba_config.CUDA_ENABLE_NRT = original_value
