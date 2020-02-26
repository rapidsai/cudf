from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from rmm._lib.device_buffer cimport device_buffer
from cudf._libxx.lib cimport (
    size_type,
    column,
    table,
    scalar,
    aggregation,
    column_contents
)

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[column] move(unique_ptr[column])
    cdef unique_ptr[table] move(unique_ptr[table])
    cdef unique_ptr[aggregation] move(unique_ptr[aggregation])
    cdef vector[unique_ptr[column]] move(vector[unique_ptr[column]])
    cdef pair[unique_ptr[table], vector[size_type]] move(
        pair[unique_ptr[table], vector[size_type]])
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
    cdef unique_ptr[scalar] move(unique_ptr[scalar])
    cdef pair[unique_ptr[device_buffer], size_type] move(
        pair[unique_ptr[device_buffer], size_type]
    )
    cdef column_contents move(column_contents)
