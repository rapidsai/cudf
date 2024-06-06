# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.types cimport interpolation, sorted

from .column cimport Column
from .table cimport Table


cpdef Column quantile(
    Column input,
    vector[double] q,
    interpolation interp = *,
    Column ordered_indices = *,
    bint exact = *
)

cpdef Table quantiles(
    Table input,
    vector[double] q,
    interpolation interp = *,
    sorted is_input_sorted = *,
    list column_order = *,
    list null_precedence = *,
)
