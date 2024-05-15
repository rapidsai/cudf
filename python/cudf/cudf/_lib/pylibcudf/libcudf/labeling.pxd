# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/labeling/label_bins.hpp" namespace "cudf" nogil:
    ctypedef enum inclusive:
        YES "cudf::inclusive::YES"
        NO "cudf::inclusive::NO"

    cdef unique_ptr[column] label_bins (
        const column_view &input,
        const column_view &left_edges,
        inclusive left_inclusive,
        const column_view &right_edges,
        inclusive right_inclusive
    ) except +
