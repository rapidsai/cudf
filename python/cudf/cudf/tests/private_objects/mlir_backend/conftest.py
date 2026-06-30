# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-collection shims for the MLIR backend suite.

Runs at conftest import time (before the test modules import
``numba_cuda_mlir``) so the workarounds below are in place by the time
collection imports the backend.
"""

from __future__ import annotations

import warnings

import numpy as np

# numba-cuda-mlir 0.4.0 has a few numpy < 2.0 compatibility issues
# TODO: remove with a suitable numba-cuda-mlir release
if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from numba_cuda_mlir.numba_cuda import types as _ncm_types

            if not hasattr(_ncm_types, "bool"):
                _ncm_types.bool = _ncm_types.bool_
                if "bool" not in _ncm_types.__all__:
                    _ncm_types.__all__.append("bool")

            import numba_cuda_mlir.cuda
            import numba_cuda_mlir.types  # noqa: F401
        except ImportError:
            pass
