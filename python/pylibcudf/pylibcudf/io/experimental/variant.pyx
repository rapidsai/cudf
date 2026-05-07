# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.variant cimport (
    cast_variant as cpp_cast_variant,
    extract_variant_field as cpp_extract_variant_field,
    get_variant_field as cpp_get_variant_field,
)
from pylibcudf.types cimport DataType
from pylibcudf.utils cimport _get_memory_resource, _get_stream

__all__ = [
    "cast_variant",
    "extract_variant_field",
    "get_variant_field",
]


cpdef Column get_variant_field(
    Column variant_column,
    str field_name,
    object stream=None,
    object mr=None,
):
    """Extract the raw VARIANT-encoded bytes of a top-level field.

    Returns a new VARIANT struct column (``struct<list<uint8>, list<uint8>>``)
    where child 0 is the input ``metadata`` (copied) and child 1 contains the
    raw encoded bytes of the named field's value for each row. Null is produced
    when the struct row is null, the field is absent, or the value blob is not
    an object.

    For details, see :cpp:func:`cudf::io::parquet::experimental::get_variant_field`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization) with ``list<uint8>`` children.
    field_name : str
        UTF-8 field name (case-sensitive).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device
        memory.

    Returns
    -------
    Column
        VARIANT struct column with the extracted field's encoded bytes.
    """
    cdef unique_ptr[column] c_result
    cdef string c_field_name = field_name.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_get_variant_field(
            variant_column.view(), c_field_name, _cs, _mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, _mr)


cpdef Column cast_variant(
    Column variant_column,
    DataType desired_type,
    object stream=None,
    object mr=None,
):
    """Decode a VARIANT struct column's ``value`` blobs into a typed column.

    Each row's ``value`` child is interpreted as a Variant-encoded primitive
    and decoded into ``desired_type``. Only ``INT32`` and ``STRING`` are
    currently supported.

    For details, see :cpp:func:`cudf::io::parquet::experimental::cast_variant`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization) with ``list<uint8>`` children.
    desired_type : DataType
        Target cuDF type (``STRING`` or ``INT32``).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device
        memory.

    Returns
    -------
    Column
        Typed column decoded from the VARIANT value blobs.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_cast_variant(
            variant_column.view(), desired_type.c_obj, _cs, _mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, _mr)


cpdef Column extract_variant_field(
    Column variant_column,
    str field_name,
    DataType desired_type,
    object stream=None,
    object mr=None,
):
    """Extract a top-level VARIANT field and decode it into a typed column.

    Convenience wrapper around ``get_variant_field`` followed by
    ``cast_variant``.

    For details, see
    :cpp:func:`cudf::io::parquet::experimental::extract_variant_field`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization).
    field_name : str
        UTF-8 field name (case-sensitive).
    desired_type : DataType
        Target type (``STRING`` or ``INT32``).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device
        memory.

    Returns
    -------
    Column
        Column of ``desired_type`` with one row per struct row; null where the
        struct row is null, the field is missing, or the encoded value does not
        match ``desired_type``.
    """
    cdef unique_ptr[column] c_result
    cdef string c_field_name = field_name.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_extract_variant_field(
            variant_column.view(),
            c_field_name,
            desired_type.c_obj,
            _cs,
            _mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), _stream, _mr)
