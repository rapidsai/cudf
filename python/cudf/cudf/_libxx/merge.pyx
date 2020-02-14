from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from libcpp.vector cimport vector
cimport cudf._libxx.includes.merge as cpp_merge


def merge_sorted(
    tables, keys=None, index=False, ascending=True, na_position="last"
):
    cdef vector[size_type] c_column_keys
    cdef vector[table_view] c_input_tables
    cdef vector[order] c_column_order
    cdef vector[null_order] c_null_precedence
    cdef order column_order
    cdef null_order null_precedence
    cdef Table source_table

    # Create vector of tables
    # Use metadata from 0th table for names, etc
    for source_table in tables:
        c_input_tables.push_back(source_table.view())
    source_table = tables[0]

    # Define sorting order and null precedence
    column_order = order.ASCENDING if ascending else order.DESCENDING
    null_precedence = (
        null_order.AFTER if na_position == "last" else null_order.BEFORE
    )

    # Define C vectors for each key column
    num_index_columns = (
        0 if source_table._index is None
        else source_table._index._num_columns
    )
    if not index and keys is not None:
        for name in keys:
            c_column_keys.push_back(
                num_index_columns + source_table._column_names.index(name)
            )
            c_column_order.push_back(column_order)
            c_null_precedence.push_back(null_precedence)
    else:
        if index:
            start=0
            stop=num_index_columns
        else:
            start=num_index_columns
            stop=num_index_columns + source_table._num_columns
        for i in range(start, stop):
            c_column_keys.push_back(i)
            c_column_order.push_back(column_order)
            c_null_precedence.push_back(null_precedence)

    # Perform sorted merge operation
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_merge.merge(
                c_input_tables,
                c_column_keys,
                c_column_order,
                c_null_precedence,
            )
        )

    # Return libxx table
    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index
            is None else source_table._index_names
        )
    )
