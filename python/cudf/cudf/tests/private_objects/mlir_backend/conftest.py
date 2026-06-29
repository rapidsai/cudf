# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Test-collection shims for the MLIR backend suite.

Runs at conftest import time (before the test modules import
``numba_cuda_mlir``) so the workaround below is in place by the time
collection imports the backend.
"""
from __future__ import annotations

import numpy as np

if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    try:
        from numba_cuda_mlir.numba_cuda import types as _ncm_types
    except ImportError:
        pass
    else:
        if not hasattr(_ncm_types, "bool"):
            _ncm_types.bool = _ncm_types.bool_
            if "bool" not in _ncm_types.__all__:
                _ncm_types.__all__.append("bool")
