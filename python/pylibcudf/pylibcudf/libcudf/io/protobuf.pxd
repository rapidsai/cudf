# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport type_id
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/io/protobuf.hpp" namespace "cudf::io::protobuf" nogil:

    cpdef enum class proto_encoding(int):
        DEFAULT
        FIXED
        ZIGZAG
        ENUM_STRING

    cpdef enum class proto_wire_type(int):
        VARINT
        I64BIT
        LEN
        SGROUP
        EGROUP
        I32BIT

    cdef struct nested_field_descriptor:
        int field_number
        int parent_idx
        int depth
        proto_wire_type wire_type
        type_id output_type
        proto_encoding encoding
        bool is_repeated
        bool is_required
        bool has_default_value

    cdef struct decode_protobuf_options:
        vector[nested_field_descriptor] schema
        vector[int64_t] default_ints
        vector[double] default_floats
        vector[bool] default_bools
        # Note: host_vector types are not easily bindable through Cython.
        # The Python layer will need to handle conversion.
        bool fail_on_errors

    cdef unique_ptr[column] decode_protobuf(
        column_view binary_input,
        decode_protobuf_options options,
        cuda_stream_view stream,
        device_memory_resource* mr,
    ) except +libcudf_exception_handler
