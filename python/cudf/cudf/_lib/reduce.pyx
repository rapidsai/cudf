# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf._lib.cpp.reduce cimport cpp_reduce, cpp_scan, scan_type
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type, type_id
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column
from cudf._lib.scalar cimport Scalar
from cudf._lib.column cimport Column
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.aggregation cimport make_aggregation, aggregation
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
import numpy as np


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
    if reduction_op in ['sum', 'sum_of_squares', 'product']:
        col_dtype = np.find_common_type([col_dtype], [np.uint64])
    col_dtype = col_dtype if dtype is None else dtype

    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    cdef unique_ptr[aggregation] c_agg = move(make_aggregation(
        reduction_op, kwargs
    ))
    cdef type_id tid = (
        <type_id> (
            <underlying_type_t_type_id> (
                np_to_cudf_types[np.dtype(col_dtype)]
            )
        )
    )

    cdef data_type c_out_dtype = data_type(tid)

    # check empty case
    if len(incol) <= incol.null_count:
        if reduction_op == 'sum' or reduction_op == 'sum_of_squares':
            return incol.dtype.type(0)
        if reduction_op == 'product':
            return incol.dtype.type(1)

        return cudf.utils.dtypes._get_nan_for_dtype(col_dtype)

    with nogil:
        c_result = move(cpp_reduce(
            c_incol_view,
            c_agg,
            c_out_dtype
        ))

    py_result = Scalar.from_unique_ptr(move(c_result))
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
