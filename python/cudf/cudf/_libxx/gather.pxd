from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *
from cudf._libxx.table cimport *


cdef extern from "cudf/copying.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] gather(table_view source_table,
                                  column_view gather_map)
