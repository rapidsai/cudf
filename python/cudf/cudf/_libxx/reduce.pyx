from cudf._libxx.cpp.reduce cimport cpp_reduce, cpp_scan, scan_type
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.types cimport data_type, type_id
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.column cimport Column
from cudf._libxx.types import np_to_cudf_types
from cudf._libxx.move cimport move
from cudf._libxx.aggregation cimport get_aggregation, aggregation
from libcpp.memory cimport unique_ptr
import numpy as np

cpdef reduce(reduction_op, Column incol, kwargs=None, dtype=None, ddof=1):
    kwargs = {'ddof': ddof}

    col_dtype = incol.dtype
    if reduction_op in ['sum', 'sum_of_squares', 'product']:
        col_dtype = np.find_common_type([col_dtype], [np.int64])
    col_dtype = col_dtype if dtype is None else dtype

    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    cdef unique_ptr[aggregation] c_agg = move(get_aggregation(
        reduction_op, kwargs
    ))
    cdef type_id tid = np_to_cudf_types[np.dtype(col_dtype)]
    cdef data_type c_out_dtype = data_type(tid)

    # check empty case
    if len(incol) <= incol.null_count:
        if reduction_op == 'sum' or reduction_op == 'sum_of_squares':
            return incol.dtype.type(0)
        if reduction_op == 'product':
            return incol.dtype.type(1)
        return np.nan

    with nogil:
        c_result = cpp_reduce(
            c_incol_view,
            c_agg,
            c_out_dtype
        )

    py_result = Scalar.from_unique_ptr(move(c_result))
    return py_result.value

cpdef scan(Column incol, scan_op, inclusive, kwargs=None):
    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[column] c_result
    cdef unique_ptr[aggregation] c_agg = move(get_aggregation(scan_op, kwargs))

    cdef scan_type c_inclusive
    if inclusive is True:
        c_inclusive = scan_type.INCLUSIVE
    elif inclusive is False:
        c_inclusive = scan_type.EXCLUSIVE

    with nogil:
        c_result = cpp_scan(
            c_incol_view,
            c_agg,
            c_inclusive
        )

    py_result = Column.from_unique_ptr(move(c_result))
    return py_result
