# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/utilities/default_stream.hpp" namespace "cudf" nogil:
    cdef bool is_ptds_enabled()
    cdef cuda_stream_view get_default_stream()
