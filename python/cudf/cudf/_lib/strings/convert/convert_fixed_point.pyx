# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.types cimport dtype_to_pylibcudf_type

import pylibcudf as plc


@acquire_spill_lock()
def from_decimal(Column input_col):
    """
    Converts a `Decimal64Column` to a `StringColumn`.

    Parameters
    ----------
    input_col : input column of type decimal

    Returns
    -------
    A column of strings representing the input decimal values.
    """
    plc_column = plc.strings.convert.convert_fixed_point.from_fixed_point(
        input_col.to_pylibcudf(mode="read"),
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def to_decimal(Column input_col, object out_type):
    """
    Returns a `Decimal64Column` from the provided `StringColumn`
    using the scale in the `out_type`.

    Parameters
    ----------
    input_col : input column of type string
    out_type : The type and scale of the decimal column expected

    Returns
    -------
    A column of decimals parsed from the string values.
    """
    plc_column = plc.strings.convert.convert_fixed_point.to_fixed_point(
        input_col.to_pylibcudf(mode="read"),
        dtype_to_pylibcudf_type(out_type),
    )
    result = Column.from_pylibcudf(plc_column)
    result.dtype.precision = out_type.precision
    return result


@acquire_spill_lock()
def is_fixed_point(Column input_col, object dtype):
    """
    Returns a Column of boolean values with True for `input_col`
    that have fixed-point characters. The output row also has a
    False value if the corresponding string would cause an integer
    overflow. The scale of the `dtype` is used to determine overflow
    in the output row.

    Parameters
    ----------
    input_col : input column of type string
    dtype : The type and scale of a decimal column

    Returns
    -------
    A Column of booleans indicating valid decimal conversion.
    """
    plc_column = plc.strings.convert.convert_fixed_point.is_fixed_point(
        input_col.to_pylibcudf(mode="read"),
        dtype_to_pylibcudf_type(dtype),
    )
    return Column.from_pylibcudf(plc_column)
