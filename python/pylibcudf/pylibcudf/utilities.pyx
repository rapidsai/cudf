# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp cimport bool
from pylibcudf.libcudf cimport utilities

__all__ = ["is_ptds_enabled"]


cpdef bool is_ptds_enabled():
    """Checks if per-thread default stream is enabled.

    For details, see :cpp:func:`is_ptds_enabled`.
    """
    return utilities.is_ptds_enabled()
