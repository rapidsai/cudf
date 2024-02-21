# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    # TODO: Eventually probably want to make this opaque, but for now it's fine
    # to treat this object as something like a POD struct
    cdef readonly:
        Py_ssize_t ptr
        object obj
