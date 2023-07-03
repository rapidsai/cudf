# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.hash cimport hash_id
from cudf._lib.cpp.types cimport size_type


cdef extern from "nvtext/minhash.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] minhash(
        const column_view &strings,
        const column_view &seeds,
        const size_type width,
        const hash_id hash_function
    ) except +
