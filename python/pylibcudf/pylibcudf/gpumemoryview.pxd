# Copyright (c) 2023-2025, NVIDIA CORPORATION.
from libc.stdint cimport uintptr_t

cdef class gpumemoryview:
    # TODO: Eventually probably want to make this opaque, but for now it's fine
    # to treat this object as something like a POD struct
    cdef readonly uintptr_t ptr
    cdef readonly object obj

    @staticmethod
    cdef gpumemoryview from_pointer(uintptr_t ptr, object owner)
