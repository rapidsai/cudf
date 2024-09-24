# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pylibcudf
from pylibcudf.null_mask import MaskState

from cudf.core.buffer import acquire_spill_lock, as_buffer

from cudf._lib.column cimport Column


@acquire_spill_lock()
def copy_bitmask(Column col):
    """
    Copies column's validity mask buffer into a new buffer, shifting by the
    offset if nonzero
    """
    if col.base_mask is None:
        return None

    rmm_db = pylibcudf.null_mask.copy_bitmask(col.to_pylibcudf(mode="read"))
    buf = as_buffer(rmm_db)
    return buf


def bitmask_allocation_size_bytes(num_bits):
    """
    Given a size, calculates the number of bytes that should be allocated for a
    column validity mask
    """
    return pylibcudf.null_mask.bitmask_allocation_size_bytes(num_bits)


def create_null_mask(size, state=MaskState.UNINITIALIZED):
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
    rmm_db = pylibcudf.null_mask.create_null_mask(size, state)
    buf = as_buffer(rmm_db)
    return buf


@acquire_spill_lock()
def bitmask_and(list columns):
    rmm_db, other = pylibcudf.null_mask.bitmask_and(
        [col.to_pylibcudf(mode="read") for col in columns]
    )
    buf = as_buffer(rmm_db)
    return buf, other


@acquire_spill_lock()
def bitmask_or(list columns):
    rmm_db, other = pylibcudf.null_mask.bitmask_or(
        [col.to_pylibcudf(mode="read") for col in columns]
    )
    buf = as_buffer(rmm_db)
    return buf, other
