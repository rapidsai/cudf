# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/minhash.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] minhash(
        const column_view &strings,
        const column_view &seeds,
        const size_type width,
    ) except +

    cdef unique_ptr[column] minhash64(
        const column_view &strings,
        const column_view &seeds,
        const size_type width,
    ) except +
