# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.libcudf.utilities cimport default_stream

__all__ = ["is_ptds_enabled"]


cpdef bool is_ptds_enabled():
    """Checks if per-thread default stream is enabled.

    For details, see :cpp:func:`is_ptds_enabled`.
    """
    return default_stream.is_ptds_enabled()
