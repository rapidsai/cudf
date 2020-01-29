import pandas as pd

from cudf._libxx.column cimport *
from cudf._libxx.table cimport *

cimport cudf._libxx.lib as libcudf


def gather(Table source_table, Column gather_map, bool check_bounds=True):
    assert pd.api.types.is_integer_dtype(gather_map.dtype)
    cdef unique_ptr[table] c_result = move(
        cpp_gather(
            source_table.view(),
            gather_map.view(),
            check_bounds
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index_names
    )


def scatter(Table source_table, Column scatter_map,
            Table target_table, bool check_bounds=True):
    assert pd.api.types.is_integer_dtype(scatter_map.dtype)
    cdef unique_ptr[table] c_result = move(
        cpp_scatter(
            source_table.view(),
            scatter_map.view(),
            target_table.data_view(),
            check_bounds
        )
    )
    result = Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names
    )
    result._index = source_table._index
    return result
