# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the numba_cuda_mlir UDF backend scaffolding (PR 1).

This PR introduces a shell of the future MLIR backend without any code
that depends on ``numba_cuda_mlir``. Tests here cover only the
infrastructure pieces:

* the ``cudf.core.udf.mlir_backend`` package is importable as an empty
  scaffold,
* ``cudf.utils._numba`` defaults ``numba.cuda.config.CUDA_ENABLE_NRT`` to
  ``False`` at import time,
* the existing ``nrt_enabled()`` and ``CaptureNRTUsage`` helpers in
  ``cudf.core.udf.nrt_utils`` behave as documented.

Subsequent PRs will add real typing/lowering tests under this directory
(e.g. ``test_strings_typing.py``, ``test_masked_lowering.py``).
"""

from __future__ import annotations

import importlib

import pytest
from numba.cuda import config as numba_config

from cudf.core.udf.nrt_utils import (
    CaptureNRTUsage,
    _current_nrt_context,
    nrt_enabled,
)


def test_mlir_backend_package_importable():
    """The empty mlir_backend package can be imported without side effects."""
    mod = importlib.import_module("cudf.core.udf.mlir_backend")
    assert mod.__doc__ and "MLIR" in mod.__doc__
    assert not hasattr(mod, "MaskedType"), (
        "PR 1 should ship an empty mlir_backend; later PRs add types here."
    )


def test_cuda_enable_nrt_default_false():
    """`cudf.utils._numba` should default `CUDA_ENABLE_NRT` to False on import."""
    import cudf.utils._numba  # noqa: F401  -- import for its side effect

    assert numba_config.CUDA_ENABLE_NRT is False, (
        "Expected CUDA_ENABLE_NRT False after `cudf.utils._numba` import; "
        "kernels that need NRT should opt in via `nrt_enabled()`."
    )


def test_nrt_enabled_round_trips_to_false():
    """`nrt_enabled()` flips the global on, then restores it."""
    import cudf.utils._numba  # noqa: F401

    assert numba_config.CUDA_ENABLE_NRT is False
    with nrt_enabled():
        assert numba_config.CUDA_ENABLE_NRT is True
    assert numba_config.CUDA_ENABLE_NRT is False


def test_nrt_enabled_restores_pre_existing_true_value():
    """If a caller manually sets the flag True, exiting the context preserves it."""
    saved = numba_config.CUDA_ENABLE_NRT
    try:
        numba_config.CUDA_ENABLE_NRT = True
        with nrt_enabled():
            assert numba_config.CUDA_ENABLE_NRT is True
        # Should restore to True (pre-context value), not to False.
        assert numba_config.CUDA_ENABLE_NRT is True
    finally:
        numba_config.CUDA_ENABLE_NRT = saved


def test_nrt_enabled_restores_on_exception():
    """An exception inside `nrt_enabled()` must still restore the prior value."""
    import cudf.utils._numba  # noqa: F401

    assert numba_config.CUDA_ENABLE_NRT is False
    with pytest.raises(RuntimeError, match="boom"):
        with nrt_enabled():
            assert numba_config.CUDA_ENABLE_NRT is True
            raise RuntimeError("boom")
    assert numba_config.CUDA_ENABLE_NRT is False


def test_capture_nrt_usage_default_false():
    """`CaptureNRTUsage` starts with `use_nrt = False`."""
    cap = CaptureNRTUsage()
    assert cap.use_nrt is False


def test_capture_nrt_usage_observes_inner_set():
    """Code inside the `with` block can flip `use_nrt` and the captor sees it."""
    with CaptureNRTUsage() as cap:
        # A type instantiation that needs NRT would do this internally:
        ctx = _current_nrt_context.get(None)
        assert ctx is cap
        ctx.use_nrt = True
    assert cap.use_nrt is True


def test_capture_nrt_usage_context_var_unset_outside():
    """The context var is reset on exit so nested code can't observe a stale captor."""
    with CaptureNRTUsage():
        pass
    assert _current_nrt_context.get(None) is None


def test_capture_nrt_usage_nested():
    """Nesting two captors restores the outer context on inner exit."""
    with CaptureNRTUsage() as outer:
        with CaptureNRTUsage() as inner:
            assert _current_nrt_context.get(None) is inner
            inner.use_nrt = True
        assert _current_nrt_context.get(None) is outer
    assert outer.use_nrt is False
    assert inner.use_nrt is True
