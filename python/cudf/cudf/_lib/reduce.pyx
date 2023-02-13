# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from cython.operator import dereference

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.aggregation cimport (
    ReduceAggregation,
    ScanAggregation,
    make_reduce_aggregation,
    make_scan_aggregation,
)
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.reduce cimport cpp_minmax, cpp_reduce, cpp_scan, scan_type
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.types cimport dtype_to_data_type, is_decimal_type_id


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

    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    cdef ReduceAggregation cython_agg = make_reduce_aggregation(
        reduction_op, kwargs)

    cdef data_type c_out_dtype = dtype_to_data_type(col_dtype)

    # check empty case
    if len(incol) <= incol.null_count:
        if reduction_op == 'sum' or reduction_op == 'sum_of_squares':
            return incol.dtype.type(0)
        if reduction_op == 'product':
            return incol.dtype.type(1)
        if reduction_op == "any":
            return False

        return cudf.utils.dtypes._get_nan_for_dtype(col_dtype)

    with nogil:
        c_result = move(cpp_reduce(
            c_incol_view,
            dereference(cython_agg.c_obj),
            c_out_dtype
        ))

    if is_decimal_type_id(c_result.get()[0].type().id()):
        scale = -c_result.get()[0].type().scale()
        precision = _reduce_precision(col_dtype, reduction_op, len(incol))
        py_result = DeviceScalar.from_unique_ptr(
            move(c_result), dtype=col_dtype.__class__(precision, scale)
        )
    else:
        py_result = DeviceScalar.from_unique_ptr(move(c_result))
    return py_result.value


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
    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[column] c_result
    cdef ScanAggregation cython_agg = make_scan_aggregation(scan_op, kwargs)

    cdef scan_type c_inclusive = \
        scan_type.INCLUSIVE if inclusive else scan_type.EXCLUSIVE

    with nogil:
        c_result = move(cpp_scan(
            c_incol_view,
            dereference(cython_agg.c_obj),
            c_inclusive
        ))

    py_result = Column.from_unique_ptr(move(c_result))
    return py_result


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
    cdef column_view c_incol_view = incol.view()
    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] c_result

    with nogil:
        c_result = move(cpp_minmax(c_incol_view))

    py_result_min = DeviceScalar.from_unique_ptr(move(c_result.first))
    py_result_max = DeviceScalar.from_unique_ptr(move(c_result.second))

    return (
        cudf.Scalar.from_device_scalar(py_result_min),
        cudf.Scalar.from_device_scalar(py_result_max)
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
