# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport interpolation, sorted
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table


cpdef Column quantile(
    Column input,
    vector[double] q,
    interpolation interp = *,
    Column ordered_indices = *,
    bint exact = *,
    Stream stream = *
)

cpdef Table quantiles(
    Table input,
    vector[double] q,
    interpolation interp = *,
    sorted is_input_sorted = *,
    list column_order = *,
    list null_precedence = *,
    Stream stream = *
)
