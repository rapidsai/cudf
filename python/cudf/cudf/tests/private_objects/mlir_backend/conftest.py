# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-collection shims for the MLIR backend suite."""

from __future__ import annotations

import warnings

import numpy as np

collect_ignore_glob: list[str] = []

# numba-cuda-mlir 0.4.0 cannot be imported under numpy < 2.0: it accesses the
# removed ``np.bool`` alias at import time (raises AttributeError) and gates its
# ``bool`` type export on numpy >= 2.0. Suppressing warnings can't help -- the
# AttributeError is unconditional -- so skip the MLIR backend tests on numpy < 2
# instead of erroring at collection.
# TODO: remove once a numba-cuda-mlir release > 0.4.0 supports numpy < 2.0.
if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import numba_cuda_mlir.cuda  # noqa: F401
    except Exception:
        collect_ignore_glob = ["test_masked_*.py"]
