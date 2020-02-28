from cudf._libxx.includes.reduce cimport cpp_reduce
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.lib cimport *
from cudf._libxx.lib import np_to_cudf_types
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column, column_view
from cudf._libxx.aggregation cimport get_aggregation, aggregation

cpdef reduce(reduction_op, Column incol, kwargs=None, dtype=None, ddof=1):
    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    print(reduction_op)
    cdef unique_ptr[aggregation] c_agg = move(get_aggregation(reduction_op, kwargs))
    cdef type_id tid = np_to_cudf_types[incol.dtype]
    cdef data_type c_out_dtype = data_type(tid)    
    with nogil:
        c_result = cpp_reduce(
            c_incol_view,
            c_agg,
            c_out_dtype
        )

    py_result = Scalar.from_unique_ptr(move(c_result))
    return py_result.value

cpdef scan(Column incol, scan_op, bool inclusive, kwargs=None):
    cdef column_view c_incol_view = incol.view()
    cdef unique_ptr[scalar] c_result
    cdef unique_ptr[aggregation] c_agg = move(get_aggregation(scan_op, kwargs))

