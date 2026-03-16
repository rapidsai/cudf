# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool


cdef extern from "cudf/context.hpp" namespace "cudf" nogil:
    void enable_jit_cache(bool enable) except +
    void clear_jit_cache() except +
