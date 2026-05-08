# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.string_view cimport string_view
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
    str path,
    object stream=None,
    object mr=None,
):
    """Extract the raw VARIANT-encoded bytes of a value at a JSONPath-like path.

    Returns a new VARIANT struct column (``struct<list<uint8>, list<uint8>>``)
    where child 0 is the input ``metadata`` (copied) and child 1 contains the
    raw encoded bytes of the value at the end of ``path``. Null is produced
    when the row is null, a name step is missing, an array index is out of
    bounds, or a step's kind does not match the current value's type.

    For details, see :cpp:func:`cudf::io::parquet::experimental::get_variant_field`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization) with ``list<uint8>`` children.
    path : str
        JSONPath-like path (e.g. ``"x"``, ``"$.foo.bar"``, ``"$[0].name"``).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device
        memory.

    Returns
    -------
    Column
        VARIANT struct column with the extracted value's encoded bytes.
    """
    cdef unique_ptr[column] c_result
    cdef bytes c_path_bytes = path.encode()
    # Hold a backing string so the string_view stays alive across the call.
    cdef string c_path = c_path_bytes
    cdef string_view c_path_view = string_view(c_path.data(), c_path.size())
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_get_variant_field(
            variant_column.view(), c_path_view, _cs, _mr.get_mr()
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
    and decoded into ``desired_type``. Supported targets: ``INT8``, ``INT16``,
    ``INT32``, ``INT64``, ``STRING``. Type matching is strict.

    For details, see :cpp:func:`cudf::io::parquet::experimental::cast_variant`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization) with ``list<uint8>`` children.
    desired_type : DataType
        Target cuDF type (``STRING`` or ``INT8``/``INT16``/``INT32``/``INT64``).
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
    str path,
    DataType desired_type,
    object stream=None,
    object mr=None,
):
    """Extract a VARIANT value at ``path`` and decode it into a typed column.

    Convenience wrapper around ``get_variant_field`` followed by
    ``cast_variant``.

    For details, see
    :cpp:func:`cudf::io::parquet::experimental::extract_variant_field`.

    Parameters
    ----------
    variant_column : Column
        Struct column (VARIANT materialization).
    path : str
        JSONPath-like path (e.g. ``"x"``, ``"$.foo.bar"``).
    desired_type : DataType
        Target type (``STRING`` or ``INT8``/``INT16``/``INT32``/``INT64``).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device
        memory.

    Returns
    -------
    Column
        Column of ``desired_type`` with one row per struct row; null where the
        struct row is null, any step along the path misses, or the encoded
        value does not match ``desired_type``.
    """
    cdef unique_ptr[column] c_result
    cdef bytes c_path_bytes = path.encode()
    cdef string c_path = c_path_bytes
    cdef string_view c_path_view = string_view(c_path.data(), c_path.size())
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_extract_variant_field(
            variant_column.view(),
            c_path_view,
            desired_type.c_obj,
            _cs,
            _mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), _stream, _mr)
