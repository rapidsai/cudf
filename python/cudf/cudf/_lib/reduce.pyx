# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import cudf
from cudf.utils.dtypes import is_decimal_dtype
from cudf.core.dtypes import Decimal64Dtype
from cudf._lib.cpp.reduce cimport cpp_reduce, cpp_scan, scan_type, cpp_minmax
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type, type_id
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.column cimport Column
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id, dtype_to_data_type
from cudf._lib.aggregation cimport make_aggregation, aggregation
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair
import numpy as np

cimport cudf._lib.cpp.types as libcudf_types


def reduce(reduction_op, Column incol, dtype=None, **kwargs):
    """
    Top level Cython reduce function wrapping libcudf++ reductions.

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

    col_dtype = incol.dtype
    if (
        reduction_op in ['sum', 'sum_of_squares', 'product']
        and not is_decimal_dtype(col_dtype)
    ):
        col_dtype = np.find_common_type([col_dtype], [np.uint64])
    col_dtype = col_dtype if dtype is None else dtype

    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    cdef unique_ptr[aggregation] c_agg = move(make_aggregation(
        reduction_op, kwargs
    ))

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
            c_agg,
            c_out_dtype
        ))

    if c_result.get()[0].type().id() == libcudf_types.type_id.DECIMAL64:
        scale = -c_result.get()[0].type().scale()
        precision = _reduce_precision(col_dtype, reduction_op, len(incol))
        py_result = DeviceScalar.from_unique_ptr(
            move(c_result), dtype=Decimal64Dtype(precision, scale)
        )
    else:
        py_result = DeviceScalar.from_unique_ptr(move(c_result))
    return py_result.value


def scan(scan_op, Column incol, inclusive, **kwargs):
    """
    Top level Cython scan function wrapping libcudf++ scans.

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
    cdef unique_ptr[aggregation] c_agg = move(
        make_aggregation(scan_op, kwargs)
    )

    cdef scan_type c_inclusive
    if inclusive is True:
        c_inclusive = scan_type.INCLUSIVE
    elif inclusive is False:
        c_inclusive = scan_type.EXCLUSIVE

    with nogil:
        c_result = move(cpp_scan(
            c_incol_view,
            c_agg,
            c_inclusive
        ))

    py_result = Column.from_unique_ptr(move(c_result))
    return py_result


def minmax(Column incol):
    """
    Top level Cython minmax function wrapping libcudf++ minmax.

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

    return cudf.Scalar(py_result_min), cudf.Scalar(py_result_max)


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
    return max(min(new_p, Decimal64Dtype.MAX_PRECISION), 0)
