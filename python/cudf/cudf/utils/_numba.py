# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numba
from numba.cuda import config as numba_config
from packaging import version

# Default NRT off for all numba-CUDA compilations driven through cudf. The
# UDF compile path (`cudf.core.udf.nrt_utils.nrt_enabled`) flips this to True
# per-kernel for UDFs whose data model declares an NRT meminfo (e.g. string
# returns), and restores it afterwards. Setting the global default to False
# avoids linking NRT into kernels that don't need it.
numba_config.CUDA_ENABLE_NRT = False


# Avoids using contextlib.contextmanager due to additional overhead
class _CUDFNumbaConfig:
    def __enter__(self) -> None:
        self.CUDA_LOW_OCCUPANCY_WARNINGS = (
            numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self.is_numba_lt_061 = version.parse(
            numba.__version__
        ) < version.parse("0.61")

        if self.is_numba_lt_061:
            self.CAPTURED_ERRORS = numba_config.CAPTURED_ERRORS
            numba_config.CAPTURED_ERRORS = "new_style"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
            self.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        if self.is_numba_lt_061:
            numba_config.CAPTURED_ERRORS = self.CAPTURED_ERRORS
