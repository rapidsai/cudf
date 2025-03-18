# Copyright (c) 2023-2025, NVIDIA CORPORATION.


cdef class gpumemoryview:
    # TODO: Eventually probably want to make this opaque, but for now it's fine
    # to treat this object as something like a POD struct
    cdef Py_ssize_t ptr
    cdef object obj

    @staticmethod
    cdef gpumemoryview from_pointer(Py_ssize_t ptr, object owner)
