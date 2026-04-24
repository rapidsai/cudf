# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.utils cimport _get_memory_resource, _get_stream

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.protobuf cimport (
    decode_protobuf as cpp_decode_protobuf,
    decode_protobuf_options,
    nested_field_descriptor,
    proto_encoding,
    proto_wire_type,
)
from pylibcudf.libcudf.types cimport type_id

__all__ = [
    "decode_protobuf",
]


cpdef Column decode_protobuf(
    Column binary_input,
    list schema,
    list default_ints,
    list default_floats,
    list default_bools,
    list default_strings,
    list enum_valid_values,
    list enum_names,
    bint fail_on_errors,
    Stream stream = None,
    DeviceMemoryResource mr = None,
):
    """
    Decode serialized protobuf messages from a LIST<UINT8> column into a STRUCT column.

    Parameters
    ----------
    binary_input : Column
        LIST<INT8/UINT8> column of serialized protobuf messages.
    schema : list of tuples
        Each tuple is (field_number, parent_idx, depth, wire_type, output_type_id,
        encoding, is_repeated, is_required, has_default_value).
    default_ints : list of int
        Default integer values per field.
    default_floats : list of float
        Default float values per field.
    default_bools : list of bool
        Default boolean values per field.
    default_strings : list of bytes
        Default string values per field (as raw bytes).
    enum_valid_values : list of list of int
        Valid enum numbers per field.
    enum_names : list of list of bytes
        UTF-8 enum names per field.
    fail_on_errors : bool
        If True, raise on malformed messages. If False, return nulls.
    stream : Stream, optional
        CUDA stream for device operations.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    Column
        A STRUCT column containing decoded protobuf fields.
    """
    cdef decode_protobuf_options options
    cdef int n = len(schema)

    # Build schema vector
    cdef vector[nested_field_descriptor] c_schema
    c_schema.reserve(n)
    cdef nested_field_descriptor desc
    for s in schema:
        desc.field_number = <int>s[0]
        desc.parent_idx = <int>s[1]
        desc.depth = <int>s[2]
        desc.wire_type = <proto_wire_type>(<int>s[3])
        desc.output_type = <type_id>(<int>s[4])
        desc.encoding = <proto_encoding>(<int>s[5])
        desc.is_repeated = <bool>s[6]
        desc.is_required = <bool>s[7]
        desc.has_default_value = <bool>s[8]
        c_schema.push_back(desc)

    options.schema = move(c_schema)

    # Default values
    cdef vector[int64_t] c_default_ints
    c_default_ints.reserve(n)
    for v in default_ints:
        c_default_ints.push_back(<int64_t>v)
    options.default_ints = move(c_default_ints)

    cdef vector[double] c_default_floats
    c_default_floats.reserve(n)
    for v in default_floats:
        c_default_floats.push_back(<double>v)
    options.default_floats = move(c_default_floats)

    cdef vector[bool] c_default_bools
    c_default_bools.reserve(n)
    for v in default_bools:
        c_default_bools.push_back(<bool>v)
    options.default_bools = move(c_default_bools)

    options.fail_on_errors = fail_on_errors

    cdef Stream s = _get_stream(stream)
    mr = _get_memory_resource(mr)

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_decode_protobuf(
                binary_input.view(),
                options,
                s.view(),
                mr.get_mr(),
            )
        )

    return Column.from_libcudf(move(c_result), s, mr)
