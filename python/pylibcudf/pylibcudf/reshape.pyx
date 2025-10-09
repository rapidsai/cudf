# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.limits cimport numeric_limits
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile,
    table_to_array as cpp_table_to_array,
    byte,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type

from pylibcudf.libcudf.utilities.span cimport device_span

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource

__all__ = ["interleave_columns", "tile", "table_to_array"]

cpdef Column interleave_columns(
    Table source_table, Stream stream=None, DeviceMemoryResource mr=None
):
    """Interleave columns of a table into a single column.

    Converts the column major table `input` into a row major column.

    Example:
    in     = [[A1, A2, A3], [B1, B2, B3]]
    return = [A1, B1, A2, B2, A3, B3]

    For details, see :cpp:func:`interleave_columns`.

    Parameters
    ----------
    source_table: Table
        The input table to interleave
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        A new column which is the result of interleaving the input columns
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_interleave_columns(
            source_table.view(), stream.view(), mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Table tile(
    Table source_table,
    size_type count,
    Stream stream=None,
    DeviceMemoryResource mr=None
):
    """Repeats the rows from input table count times to form a new table.

    For details, see :cpp:func:`tile`.

    Parameters
    ----------
    source_table: Table
        The input table containing rows to be repeated
    count: size_type
        The number of times to tile "rows". Must be non-negative
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Table
        The table containing the tiled "rows"
    """
    cdef unique_ptr[table] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_tile(
            source_table.view(), count, stream.view(), mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), stream, mr)


cpdef void table_to_array(
    Table input_table,
    uintptr_t ptr,
    size_t size,
    Stream stream=None
):
    """
    Copy a table into a preallocated column-major device array.

    Parameters
    ----------
    input_table : Table
        A table with fixed-width, non-nullable columns of the same type.
    ptr : uintptr_t
        A device pointer to the beginning of the output buffer.
    size : size_t
        The total number of bytes available at `ptr`.
        Must be at least `num_rows * num_columns * sizeof(dtype)`.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    """
    if size > numeric_limits[size_t].max():
        raise ValueError(
            "Size exceeds the size_t limit."
        )
    stream = _get_stream(stream)

    cdef device_span[byte] span = device_span[byte](
        <byte*> ptr, size
    )

    with nogil:
        cpp_table_to_array(
            input_table.view(),
            span,
            stream.view()
        )
