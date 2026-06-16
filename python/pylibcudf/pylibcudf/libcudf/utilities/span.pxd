# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/utilities/span.hpp" namespace "cudf" nogil:
    cdef cppclass host_span[T]:
        host_span() except +libcudf_exception_handler
        host_span(vector[T]) except +libcudf_exception_handler
        host_span(T* data, size_type size) noexcept

    cdef cppclass device_span[T]:
        device_span() noexcept
        device_span(T *data, size_type size) noexcept
        T *data() noexcept
