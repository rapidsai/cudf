from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport size_type

from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.table.table_view cimport (
    table_view,
    mutable_table_view
)

cdef extern from "cudf/table/table.hpp" namespace "cudf::experimental" nogil:
    cdef cppclass table:
        table(const table&) except +
        table(vector[unique_ptr[column]]&& columns) except +
        table(table_view) except +
        size_type num_columns() except +
        table_view view() except +
        mutable_table_view mutable_view() except +
        vector[unique_ptr[column]] release() except +
