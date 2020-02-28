# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.pair cimport pair

from cudf._libxx.column cimport Column
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer
from cudf.core.buffer import Buffer

from cudf._libxx.column cimport Column
from cudf._libxx.move cimport move

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.transform cimport bools_to_mask as cpp_bools_to_mask
from cudf._libxx.cpp.column.column_view cimport column_view
cimport cudf._libxx.cpp.transform as libcudf_transform


def bools_to_mask(Column col):
    """
    Given an int8 (boolean) column, compress the data from booleans to bits and
    return a Buffer
    """
    cdef column_view col_view = col.view()
    cdef pair[unique_ptr[device_buffer], size_type] cpp_out
    cdef unique_ptr[device_buffer] up_db
    cdef size_type null_count

    with nogil:
        cpp_out = move(libcudf_transform.bools_to_mask(col_view))
        up_db = move(cpp_out.first)
        # null_count = cpp_out.second

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = Buffer(rmm_db)
    return buf


def nans_to_nulls(Column input):
    cdef column_view c_input = input.view()
    cdef pair[unique_ptr[device_buffer], size_type] c_output
    cdef unique_ptr[device_buffer] c_buffer

    with nogil:
        c_buffer = move(libcudf_transform.nans_to_nulls(c_input).first)
        # c_buffer = move(c_output.first)

    buffer = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    buffer = Buffer(buffer)
    return buffer
