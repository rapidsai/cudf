# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import Enum

from cudf._lib import pylibcudf
from cudf.core.buffer import acquire_spill_lock, as_buffer

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.null_mask cimport underlying_type_t_mask_state
from cudf._lib.pylibcudf.libcudf.types cimport mask_state, size_type


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

    rmm_db = pylibcudf.null_mask.copy_bitmask(input_col.to_pylibcudf(mode="read"))
    buf = as_buffer(rmm_db)
    return buf


def bitmask_allocation_size_bytes(size_type num_bits):
    """
    Given a size, calculates the number of bytes that should be allocated for a
    column validity mask
    """
    return pylibcudf.null_mask.bitmask_allocation_size_bytes(num_bits)


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

    cdef mask_state c_mask_state = <mask_state>(
        <underlying_type_t_mask_state>(state.value)
    )

    rmm_db = pylibcudf.null_mask.create_null_mask(size, c_mask_state)
    buf = as_buffer(rmm_db)
    return buf


@acquire_spill_lock()
def bitmask_and(columns: list):
    rmm_db, other = pylibcudf.null_mask.bitmask_and(
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in columns]).view()
    )
    buf = as_buffer(rmm_db)
    return buf, other


@acquire_spill_lock()
def bitmask_or(columns: list):
    rmm_db, other = pylibcudf.null_mask.bitmask_or(
        pylibcudf.Table([col.to_pylibcudf(mode="read") for col in columns]).view()
    )
    buf = as_buffer(rmm_db)
    return buf, other
