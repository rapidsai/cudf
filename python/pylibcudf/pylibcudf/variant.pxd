# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool


cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant:
        variant& operator=(variant&)
        size_t index()

    cdef cppclass monostate:
        pass

    cdef T* get_if[T](...)
    cdef bool holds_alternative[T](...)
