# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.types cimport interpolation
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.column.column_view cimport column_view

cimport cudf._libxx.cpp.types as libcudf_types


cdef extern from "cudf/quantiles.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] quantiles(
        const table_view & source_table,
        const vector[double]& q,
        interpolation interp,
        const column_view & ordered_indices,
        bool cast_to_doubles
    ) except +
