from libcpp.vector cimport vector
from libcpp.utility cimport pair

from cudf.bindings.groupby.sort cimport *
from cudf.bindings.utils import *
from cudf.bindings.utils cimport *

def apply_groupby_without_aggregations(cols, key_cols):
    from cudf.dataframe.column import Column
    
    cdef cudf_table* c_in_table = table_from_columns(cols)
    cdef vector[gdf_index_type] c_key_col_indices
    cdef pair[cudf_table, gdf_column] c_result

    for i in range(len(key_cols)):
        if key_cols[i] in cols:
            c_key_col_indices.push_back(cols.index(key_cols[i]))

    cdef size_t c_num_key_cols = c_key_col_indices.size()

    cdef gdf_context* c_ctx = create_context_view(
        0,
        'sort',
        0,
        0,
        0,
        'null_as_largest',
        False
    )

    with nogil:
        c_result = gdf_group_by_without_aggregations(
            c_in_table[0],
            c_num_key_cols,
            c_key_col_indices.data(),
            c_ctx
            )
        
    data, mask = gdf_column_to_column_mem(&c_result.second)
    indices_column = Column.from_mem_views(data, mask)
    sorted_cols = columns_from_table(&c_result.first)
    del c_in_table

    return sorted_cols, indices_column
    
    
    

    
