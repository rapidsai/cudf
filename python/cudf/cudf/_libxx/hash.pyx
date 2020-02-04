from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table

def _hash_partition(source_table, columns_to_hash, num_partitions):
    cdef vector[size_type] c_columns_to_hash = columns_to_hash
    cdef int c_num_partitions = num_partitions

    cdef pair[unique_ptr[table], vector[size_type]] c_result = (
        cpp_hash_partition(
            source_table.view(),
            c_columns_to_hash,
            c_num_partitions
        )
    )

    return (Table.from_unique_ptr(move(c_result.first),
        column_names=source_table._column_names,
        index_names=source_table._index._column_names),
        list(c_result.second))

def _hash(source_table, initial_hash_values=None):
    cdef vector[uint32_t] c_initial_hash = initial_hash_values

    cdef unique_ptr[column] c_result = (
        cpp_hash(
            source_table.view(),
            c_initial_hash
        )
    )

    return Column.from_unique_ptr(move(c_result))
