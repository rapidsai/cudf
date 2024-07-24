# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp cimport bool

from cudf._lib.pylibcudf.libcudf.rolling cimport window_type
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar

ctypedef fused WindowArg:
    Column
    size_type


cpdef Column rolling_window(
    Column source,
    WindowArg preceding_window,
    WindowArg following_window,
    size_type min_periods,
    Aggregation agg,
)

cpdef tuple[Column, Column] windows_from_offset(
    Column input,
    Scalar length,
    Scalar offset,
    window_type window_type,
    bool only_preceding,
)
