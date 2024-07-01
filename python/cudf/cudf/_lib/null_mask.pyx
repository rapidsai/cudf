# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import Enum

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf.core.buffer import acquire_spill_lock, as_buffer

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.null_mask cimport (
    bitmask_allocation_size_bytes as cpp_bitmask_allocation_size_bytes,
    bitmask_and as cpp_bitmask_and,
    bitmask_or as cpp_bitmask_or,
    copy_bitmask as cpp_copy_bitmask,
    create_null_mask as cpp_create_null_mask,
    underlying_type_t_mask_state,
)
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport mask_state, size_type
from cudf._lib.utils cimport table_view_from_columns


class MaskState(Enum):
    """
    Enum for null mask creation state
    """
    UNALLOCATED = <underlying_type_t_mask_state> mask_state.UNALLOCATED
    UNINITIALIZED = <underlying_type_t_mask_state> mask_state.UNINITIALIZED
    ALL_VALID = <underlying_type_t_mask_state> mask_state.ALL_VALID
    ALL_NULL = <underlying_type_t_mask_state> mask_state.ALL_NULL


@acquire_spill_lock()
def copy_bitmask(Column col):
    """
    Copies column's validity mask buffer into a new buffer, shifting by the
    offset if nonzero
    """
    if col.base_mask is None:
        return None

    cdef column_view col_view = col.view()
    cdef device_buffer db
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        db = move(cpp_copy_bitmask(col_view))
        up_db = move(make_unique[device_buffer](move(db)))

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = as_buffer(rmm_db)
    return buf


def bitmask_allocation_size_bytes(size_type num_bits):
    """
    Given a size, calculates the number of bytes that should be allocated for a
    column validity mask
    """
    cdef size_t output_size

    with nogil:
        output_size = cpp_bitmask_allocation_size_bytes(num_bits)

    return output_size


def create_null_mask(size_type size, state=MaskState.UNINITIALIZED):
    """
    Given a size and a mask state, allocate a mask that can properly represent
    the given size with the given mask state

    Parameters
    ----------
    size : int
        Number of elements the mask needs to be able to represent
    state : ``MaskState``, default ``MaskState.UNINITIALIZED``
        State the null mask should be created in
    """
    if not isinstance(state, MaskState):
        raise TypeError(
            "`state` is required to be of type `MaskState`, got "
            + (type(state).__name__)
        )

    cdef device_buffer db
    cdef unique_ptr[device_buffer] up_db
    cdef mask_state c_mask_state = <mask_state>(
        <underlying_type_t_mask_state>(state.value)
    )

    with nogil:
        db = move(cpp_create_null_mask(size, c_mask_state))
        up_db = move(make_unique[device_buffer](move(db)))

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = as_buffer(rmm_db)
    return buf


@acquire_spill_lock()
def bitmask_and(columns: list):
    cdef table_view c_view = table_view_from_columns(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef unique_ptr[device_buffer] up_db
    with nogil:
        c_result = move(cpp_bitmask_and(c_view))
        up_db = move(make_unique[device_buffer](move(c_result.first)))
    dbuf = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = as_buffer(dbuf)
    return buf, c_result.second


@acquire_spill_lock()
def bitmask_or(columns: list):
    cdef table_view c_view = table_view_from_columns(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef unique_ptr[device_buffer] up_db
    with nogil:
        c_result = move(cpp_bitmask_or(c_view))
        up_db = move(make_unique[device_buffer](move(c_result.first)))
    dbuf = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = as_buffer(dbuf)
    return buf, c_result.second
