# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf._lib.column cimport Column
from cudf._lib.types cimport dtype_to_pylibcudf_type

import numpy as np

import pylibcudf

from cudf.api.types import is_decimal_dtype
from cudf.core.buffer import acquire_spill_lock


@acquire_spill_lock()
def unary_operation(Column input, object op):
    return Column.from_pylibcudf(
        pylibcudf.unary.unary_operation(input.to_pylibcudf(mode="read"), op)
    )


@acquire_spill_lock()
def is_null(Column input):
    return Column.from_pylibcudf(
        pylibcudf.unary.is_null(input.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def is_valid(Column input):
    return Column.from_pylibcudf(
        pylibcudf.unary.is_valid(input.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def cast(Column input, object dtype=np.float64):
    result = Column.from_pylibcudf(
        pylibcudf.unary.cast(
            input.to_pylibcudf(mode="read"),
            dtype_to_pylibcudf_type(dtype)
        )
    )

    if is_decimal_dtype(result.dtype):
        result.dtype.precision = dtype.precision
    return result


@acquire_spill_lock()
def is_nan(Column input):
    return Column.from_pylibcudf(
        pylibcudf.unary.is_nan(input.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def is_non_nan(Column input):
    return Column.from_pylibcudf(
        pylibcudf.unary.is_not_nan(input.to_pylibcudf(mode="read"))
    )
