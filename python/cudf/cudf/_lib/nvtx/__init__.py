# Copyright (c) 2020, NVIDIA CORPORATION.

import os

from cudf._lib.nvtx.nvtx import _annotate_nop, annotate, pop_range, push_range

if os.getenv("PYNVTX_DISABLE"):
    annotate = _annotate_nop
