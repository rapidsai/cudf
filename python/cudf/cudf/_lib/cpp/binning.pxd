# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

cdef extern from "cudf/binning.hpp" namespace "cudf" nogil:
    ctypedef enum inclusive:
        YES "cudf::bin::YES"
        NO "cudf::bin::NO"

    cdef unique_ptr<column> bin (
        const column_view &input,
        const column_view &left_edges,
        inclusive left_inclusive,
        const column_view &right_edges,
        inclusive right_inclusive
    ) except +
