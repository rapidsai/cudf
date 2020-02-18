# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from enum import Enum

from libcpp.memory cimport unique_ptr, make_unique

import cudf._libxx as libcudfxx
from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.includes.null_mask cimport (
    copy_bitmask as cpp_copy_bitmask,
    create_null_mask as cpp_create_null_mask,
    bitmask_allocation_size_bytes as cpp_bitmask_allocation_size_bytes
)

from cudf.core.buffer import Buffer


cdef class mask_state_value:
    """
    Wraps a ``mask_state`` enum class value in a Cython object
    """
    cdef mask_state c_value

    def __cinit__(self):
        pass

    @staticmethod
    cdef mask_state_value from_c_val(mask_state c_val):
        cdef mask_state_value wrapper = mask_state_value()
        wrapper.c_value = c_val
        return wrapper


class MaskState(Enum):
    """
    Enum for null mask creation state
    """
    UNALLOCATED = mask_state_value.from_c_val(mask_state.UNALLOCATED)
    UNINITIALIZED = mask_state_value.from_c_val(mask_state.UNINITIALIZED)
    ALL_VALID = mask_state_value.from_c_val(mask_state.ALL_VALID)
    ALL_NULL = mask_state_value.from_c_val(mask_state.ALL_NULL)


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
        db = cpp_copy_bitmask(col_view)
        up_db = make_unique[device_buffer](move(db))

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = Buffer(rmm_db)
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
    cdef mask_state_value cy_mask_state = state.value
    cdef mask_state c_mask_state = cy_mask_state.c_value

    with nogil:
        db = cpp_create_null_mask(size, c_mask_state)
        up_db = make_unique[device_buffer](move(db))

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = Buffer(rmm_db)
    return buf
