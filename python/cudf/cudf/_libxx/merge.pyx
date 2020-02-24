from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from libcpp.vector cimport vector
cimport cudf._libxx.includes.merge as cpp_merge


def merge_sorted(
    object tables,
    object keys=None,
    bool by_index=False,
    bool ignore_index=False,
    bool ascending=True,
    object na_position="last",
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
    c_input_tables.reserve(len(tables))
    for source_table in tables:
        if ignore_index:
            c_input_tables.push_back(source_table.data_view())
        else:
            c_input_tables.push_back(source_table.view())
    source_table = tables[0]

    # Define sorting order and null precedence
    column_order = order.ASCENDING if ascending else order.DESCENDING
    null_precedence = (
        null_order.BEFORE if na_position == "first" else null_order.AFTER
    )

    # Determine index-column offset and index names
    if ignore_index:
        num_index_columns = 0
        index_names = None
    else:
        num_index_columns = (
            0 if source_table._index is None
            else source_table._index._num_columns
        )
        index_names = source_table._index_names

    # Define C vectors for each key column
    if not by_index and keys is not None:
        num_keys = len(keys)
        c_column_keys.reserve(num_keys)
        for name in keys:
            c_column_keys.push_back(
                num_index_columns + source_table._column_names.index(name)
            )
    else:
        if by_index:
            start = 0
            stop = num_index_columns
        else:
            start = num_index_columns
            stop = num_index_columns + source_table._num_columns
        num_keys = stop - start
        c_column_keys.reserve(num_keys)
        for key in range(start, stop):
            c_column_keys.push_back(key)
    c_column_order = vector[order](num_keys, column_order)
    c_null_precedence = vector[null_order](num_keys, null_precedence)

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
        index_names=index_names,
    )
