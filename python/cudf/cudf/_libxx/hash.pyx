# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.includes.lib cimport *
from cudf._libxx.includes.column cimport Column
from cudf._libxx.includes.table cimport Table

from cudf._libxx.includes.hash cimport (
    hash_partition as cpp_hash_partition,
    hash as cpp_hash
)

def _hash_partition(Table source_table, columns_to_hash, num_partitions):
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

def _hash(Table source_table, initial_hash_values=None):
    cdef vector[uint32_t] c_initial_hash = initial_hash_values

    cdef unique_ptr[column] c_result = (
        cpp_hash(
            source_table.data_view(),
            c_initial_hash
        )
    )

    return Column.from_unique_ptr(move(c_result))
