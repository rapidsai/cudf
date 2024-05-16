# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/transpose.hpp" namespace "cudf" nogil:
    cdef pair[
        unique_ptr[column],
        table_view
    ] transpose(
        table_view input_table
    ) except +
