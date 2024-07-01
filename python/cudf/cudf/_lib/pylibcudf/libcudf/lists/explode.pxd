# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/lists/explode.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] explode_outer(
        const table_view,
        size_type explode_column_idx,
    ) except +
