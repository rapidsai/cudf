# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import cudf
from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.types cimport dtype_to_pylibcudf_type, is_decimal_type_id

from cudf._lib import pylibcudf
from cudf._lib.aggregation import make_aggregation


@acquire_spill_lock()
def reduce(reduction_op, Column incol, dtype=None, **kwargs):
    """
    Top level Cython reduce function wrapping libcudf reductions.

    Parameters
    ----------
    reduction_op : string
        A string specifying the operation, e.g. sum, prod
    incol : Column
        A cuDF Column object
    dtype: numpy.dtype, optional
        A numpy data type to use for the output, defaults
        to the same type as the input column
    """

    col_dtype = (
        dtype if dtype is not None
        else incol._reduction_result_dtype(reduction_op)
    )

    # check empty case
    if len(incol) <= incol.null_count:
        if reduction_op == 'sum' or reduction_op == 'sum_of_squares':
            return incol.dtype.type(0)
        if reduction_op == 'product':
            return incol.dtype.type(1)
        if reduction_op == "any":
            return False

        return cudf.utils.dtypes._get_nan_for_dtype(col_dtype)

    result = pylibcudf.reduce.reduce(
        incol.to_pylibcudf(mode="read"),
        make_aggregation(reduction_op, kwargs).c_obj,
        dtype_to_pylibcudf_type(col_dtype),
    )

    if is_decimal_type_id(result.type().id()):
        scale = -result.type().scale()
        precision = _reduce_precision(col_dtype, reduction_op, len(incol))
        return DeviceScalar.from_pylibcudf(
            result,
            dtype=col_dtype.__class__(precision, scale),
        ).value
    return DeviceScalar.from_pylibcudf(result).value


@acquire_spill_lock()
def scan(scan_op, Column incol, inclusive, **kwargs):
    """
    Top level Cython scan function wrapping libcudf scans.

    Parameters
    ----------
    incol : Column
        A cuDF Column object
    scan_op : string
        A string specifying the operation, e.g. cumprod
    inclusive: bool
        Flag for including nulls in relevant scan
    """
    return Column.from_pylibcudf(
        pylibcudf.reduce.scan(
            incol.to_pylibcudf(mode="read"),
            make_aggregation(scan_op, kwargs).c_obj,
            pylibcudf.reduce.ScanType.INCLUSIVE if inclusive
            else pylibcudf.reduce.ScanType.EXCLUSIVE,
        )
    )


@acquire_spill_lock()
def minmax(Column incol):
    """
    Top level Cython minmax function wrapping libcudf minmax.

    Parameters
    ----------
    incol : Column
        A cuDF Column object

    Returns
    -------
    A pair of ``(min, max)`` values of ``incol``
    """
    min, max = pylibcudf.reduce.minmax(incol.to_pylibcudf(mode="read"))
    return (
        cudf.Scalar.from_device_scalar(DeviceScalar.from_pylibcudf(min)),
        cudf.Scalar.from_device_scalar(DeviceScalar.from_pylibcudf(max)),
    )


def _reduce_precision(dtype, op, nrows):
    """
    Returns the result precision when performing the reduce
    operation `op` for the given dtype and column size.

    See: https://docs.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
    """  # noqa: E501
    p = dtype.precision
    if op in ("min", "max"):
        new_p = p
    elif op == "sum":
        new_p = p + nrows - 1
    elif op == "product":
        new_p = p * nrows + nrows - 1
    elif op == "sum_of_squares":
        new_p = 2 * p + nrows
    else:
        raise NotImplementedError()
    return max(min(new_p, dtype.MAX_PRECISION), 0)
