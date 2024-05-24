# Copyright (c) 2024, NVIDIA CORPORATION.

from .column cimport Column
from .table cimport Table
from .types cimport interpolation, sorted


cpdef Column quantile(
    Column input,
    const double[:] q,
    interpolation interp = *,
    Column ordered_indices = *,
    bint exact = *
)

cpdef Table quantiles(
    Table input,
    const double[:] q,
    interpolation interp = *,
    sorted is_input_sorted = *,
    list column_order = *,
    list null_precedence = *,
)
