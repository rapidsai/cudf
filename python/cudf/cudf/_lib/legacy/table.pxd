# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.legacy.cudf cimport *

cdef class TableView:
    cdef cudf_table* ptr
    cdef vector[gdf_column*] c_columns

cdef class Table:
    cdef unique_ptr[cudf_table] ptr
    cdef vector[gdf_column*] c_columns

    @staticmethod
    cdef from_ptr(unique_ptr[cudf_table]&& ptr)

cdef extern from "<utility>" namespace "std" nogil:
    cdef cudf_table move(cudf_table)
    cdef unique_ptr[cudf_table] move(unique_ptr[cudf_table])
