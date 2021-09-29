# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np

import cudf
from cudf.utils import cudautils

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table, table_view_from_table

from cudf.core.buffer import Buffer

from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id

from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.utils cimport data_from_unique_ptr

from numba.np import numpy_support

cimport cudf._lib.cpp.transform as libcudf_transform


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

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = Buffer(rmm_db)
    return buf


def mask_to_bools(object mask_buffer, size_type begin_bit, size_type end_bit):
    """
    Given a mask buffer, returns a boolean column representng bit 0 -> False
    and 1 -> True within range of [begin_bit, end_bit),
    """
    if not isinstance(mask_buffer, cudf.core.buffer.Buffer):
        raise TypeError("mask_buffer is not an instance of "
                        "cudf.core.buffer.Buffer")
    cdef bitmask_type* bit_mask = <bitmask_type*><uintptr_t>(mask_buffer.ptr)

    cdef unique_ptr[column] result
    with nogil:
        result = move(
            libcudf_transform.mask_to_bools(bit_mask, begin_bit, end_bit)
        )

    return Column.from_unique_ptr(move(result))


def nans_to_nulls(Column input):
    cdef column_view c_input = input.view()
    cdef pair[unique_ptr[device_buffer], size_type] c_output
    cdef unique_ptr[device_buffer] c_buffer

    with nogil:
        c_output = move(libcudf_transform.nans_to_nulls(c_input))
        c_buffer = move(c_output.first)

    if c_output.second == 0:
        return None

    buffer = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    buffer = Buffer(buffer)
    return buffer


def transform(Column input, op):
    cdef column_view c_input = input.view()
    cdef string c_str
    cdef type_id c_tid
    cdef data_type c_dtype

    nb_type = numpy_support.from_dtype(input.dtype)
    nb_signature = (nb_type,)
    compiled_op = cudautils.compile_udf(op, nb_signature)
    c_str = compiled_op[0].encode('UTF-8')
    np_dtype = cudf.dtype(compiled_op[1])

    try:
        c_tid = <type_id> (
            <underlying_type_t_type_id> SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[
                np_dtype
            ]
        )
        c_dtype = data_type(c_tid)

    except KeyError:
        raise TypeError(
            "Result of window function has unsupported dtype {}"
            .format(np_dtype)
        )

    with nogil:
        c_output = move(libcudf_transform.transform(
            c_input,
            c_str,
            c_dtype,
            True
        ))

    return Column.from_unique_ptr(move(c_output))


def masked_udf(Table incols, op, output_type):
    cdef table_view data_view = table_view_from_table(
        incols, ignore_index=True)
    cdef string c_str = op.encode("UTF-8")
    cdef type_id c_tid
    cdef data_type c_dtype

    c_tid = <type_id> (
        <underlying_type_t_type_id> SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[
            output_type
        ]
    )
    c_dtype = data_type(c_tid)

    with nogil:
        c_output = move(libcudf_transform.generalized_masked_op(
            data_view,
            c_str,
            c_dtype,
        ))

    return Column.from_unique_ptr(move(c_output))


def table_encode(Table input):
    cdef table_view c_input = table_view_from_table(
        input, ignore_index=True)
    cdef pair[unique_ptr[table], unique_ptr[column]] c_result

    with nogil:
        c_result = move(libcudf_transform.encode(c_input))

    return (
        *data_from_unique_ptr(
            move(c_result.first),
            column_names=input._column_names,
        ),
        Column.from_unique_ptr(move(c_result.second))
    )
