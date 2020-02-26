from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view
cimport cudf._libxx.includes.types as cudf_types


cdef extern from "cudf/merge.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] merge (
        vector[table_view] tables_to_merge,
        vector[cudf_types.size_type] key_cols,
        vector[cudf_types.order] column_order,
        vector[cudf_types.null_order] null_precedence,
    ) except +
