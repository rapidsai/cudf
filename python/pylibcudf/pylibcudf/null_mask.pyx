# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport make_unique
from libcpp.pair cimport pair
from libcpp.utility cimport move
from pylibcudf.libcudf cimport null_mask as cpp_null_mask
from pylibcudf.libcudf.types cimport mask_state, size_type, bitmask_type
from pylibcudf.gpumemoryview cimport gpumemoryview

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.libcudf.types import mask_state as MaskState  # no-cython-lint

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource

__all__ = [
    "bitmask_allocation_size_bytes",
    "bitmask_and",
    "bitmask_or",
    "copy_bitmask",
    "create_null_mask",
    "null_count",
]

cdef DeviceBuffer buffer_to_python(
    device_buffer buf, Stream stream, DeviceMemoryResource mr
):
    return DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](move(buf)), stream, mr
    )


cpdef DeviceBuffer copy_bitmask(
    Column col,
    Stream stream=None,
    DeviceMemoryResource mr=None
):
    """Copies ``col``'s bitmask into a ``DeviceBuffer``.

    For details, see :cpp:func:`copy_bitmask`.

    Parameters
    ----------
    col : Column
        Column whose bitmask needs to be copied
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    rmm.DeviceBuffer
        A ``DeviceBuffer`` containing ``col``'s bitmask, or an empty
        ``DeviceBuffer`` if ``col`` is not nullable
    """
    cdef device_buffer db
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        db = cpp_null_mask.copy_bitmask(col.view(), stream.view(), mr.get_mr())

    return buffer_to_python(move(db), stream, mr)

cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits):
    """
    Computes the required bytes necessary to represent the specified number of bits
    with a 64B padding boundary.

    For details, see :cpp:func:`bitmask_allocation_size_bytes`.

    Parameters
    ----------
    number_of_bits : size_type
        The number of bits that need to be represented

    Returns
    -------
    size_t
        The necessary number of bytes
    """
    with nogil:
        return cpp_null_mask.bitmask_allocation_size_bytes(number_of_bits, 64)


cpdef DeviceBuffer create_null_mask(
    size_type size,
    mask_state state = mask_state.UNINITIALIZED,
    Stream stream=None,
    DeviceMemoryResource mr=None
):
    """Creates a ``DeviceBuffer`` for use as a null value indicator bitmask of a
    ``Column``.

    For details, see :cpp:func:`create_null_mask`.

    Parameters
    ----------
    size : size_type
        The number of elements to be represented by the mask
    state : mask_state, optional
        The desired state of the mask. Can be one of { MaskState.UNALLOCATED,
        MaskState.UNINITIALIZED, MaskState.ALL_VALID, MaskState.ALL_NULL }
        (default MaskState.UNINITIALIZED)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    rmm.DeviceBuffer
        A ``DeviceBuffer`` for use as a null bitmask satisfying the desired size and
        state
    """
    cdef device_buffer db
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        db = cpp_null_mask.create_null_mask(size, state, stream.view(), mr.get_mr())

    return buffer_to_python(move(db), stream, mr)


cpdef tuple bitmask_and(list columns, Stream stream=None, DeviceMemoryResource mr=None):
    """Performs bitwise AND of the bitmasks of a list of columns.

    For details, see :cpp:func:`bitmask_and`.

    Parameters
    ----------
    columns : list
        The list of columns
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    tuple[DeviceBuffer, size_type]
        A tuple of the resulting mask and count of unset bits
    """
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_null_mask.bitmask_and(c_table.view(), stream.view(), mr.get_mr())

    return buffer_to_python(move(c_result.first), stream, mr), c_result.second


cpdef tuple bitmask_or(list columns, Stream stream=None, DeviceMemoryResource mr=None):
    """Performs bitwise OR of the bitmasks of a list of columns.

    For details, see :cpp:func:`bitmask_or`.

    Parameters
    ----------
    columns : list
        The list of columns
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    tuple[DeviceBuffer, size_type]
        A tuple of the resulting mask and count of unset bits
    """
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_null_mask.bitmask_or(c_table.view(), stream.view(), mr.get_mr())

    return buffer_to_python(move(c_result.first), stream, mr), c_result.second


cpdef size_type null_count(
    gpumemoryview bitmask,
    size_type start,
    size_type stop,
    Stream stream=None
):
    """Given a validity bitmask, counts the number of null elements.

    For details, see :cpp:func:`null_count`.

    Parameters
    ----------
    bitmask : int
        Integer pointer to the bitmask.
    start : int
        Index of the first bit to count (inclusive).
    stop : int
        Index of the last bit to count (exclusive).
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    int
        The number of null elements in the specified range.
    """
    stream = _get_stream(stream)
    with nogil:
        return cpp_null_mask.null_count(
            <bitmask_type*>(bitmask.ptr),
            start,
            stop,
            stream.view()
        )
