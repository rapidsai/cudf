# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .aggregation cimport Aggregation
from .column cimport Column

ctypedef fused WindowType:
    Column
    size_type


cpdef Column rolling_window(
    Column source,
    WindowType preceding_window,
    WindowType following_window,
    size_type min_periods,
    Aggregation agg,
)
