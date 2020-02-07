from cudf._libxx.includes.lib cimport *
from cudf._libxx.includes.column cimport Column
from cudf._libxx.includes.table cimport Table
from libcpp.vector cimport vector
cimport cudf._libxx.includes.merge as cpp_merge


ascending_dict = {True: ASCENDING, False: DESCENDING}
nulls_after_dict = {True: AFTER, False: BEFORE}


def sorted_merge(tables, keys, ascending=True, nulls_after=True):
    cdef vector[size_type] c_column_keys
    cdef vector[table_view] c_input_tables
    cdef unique_ptr[table] c_output_table
    cdef vector[order] c_column_order
    cdef vector[null_order] c_null_precedence
    cdef size_type key
    cdef Table input_table

    for input_table in tables:
        c_input_tables.push_back(input_table.view())

    for key in keys:
        c_column_keys.push_back(key)
        c_column_order.push_back(ascending_dict[ascending])
        c_null_precedence.push_back(nulls_after_dict[nulls_after])

    cdef unique_ptr[table] c_ouput_table = (
        cpp_merge.merge(
            c_input_tables,
            c_column_keys,
            c_column_order,
            c_null_precedence,
        )
    )

    input_table = tables[0]
    return Table.from_unique_ptr(
        move(c_output_table),
        column_names=input_table._column_names,
        index_names=input_table._index._column_names
    )
