import pandas as pd

from cudf._libxx.includes.lib cimport *
from cudf._libxx.includes.column cimport Column
from cudf._libxx.includes.table cimport Table

cdef extern from "cudf/copying.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] cpp_gather "cudf::experimental::gather" (
        table_view source_table,
        column_view gather_map
    )

def gather(Table source_table, Column gather_map):
    assert pd.api.types.is_integer_dtype(gather_map.dtype)
    cdef unique_ptr[table] c_result = (
        cpp_gather(
            source_table.view(),
            gather_map.view()
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index._column_names
    )
