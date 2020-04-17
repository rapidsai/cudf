# Copyright (c) 2020, NVIDIA CORPORATION.

import os

from cudf._lib.nvtx.nvtx import _annotate_nop, annotate, pop_range, push_range

if os.getenv("CUDF_DISABLE_NVTX"):
    annotate = _annotate_nop
