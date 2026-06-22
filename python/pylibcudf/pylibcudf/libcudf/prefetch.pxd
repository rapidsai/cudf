# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "cudf/utilities/prefetch.hpp" namespace "cudf::prefetch" nogil:
    void enable() noexcept
    void disable() noexcept
    void enable_debugging() noexcept
    void disable_debugging() noexcept
