# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.hash cimport (
    hash as cpp_hash
)
from cudf._lib.cpp.partitioning cimport (
    hash_partition as cpp_hash_partition,
)
cimport cudf._lib.cpp.types as libcudf_types


def hash_partition(Table source_table, object columns_to_hash,
                   int num_partitions, bool keep_index=True):
    cdef vector[libcudf_types.size_type] c_columns_to_hash = columns_to_hash
    cdef int c_num_partitions = num_partitions
    cdef table_view c_source_view
    if keep_index is True:
        c_source_view = source_table.view()
    else:
        c_source_view = source_table.data_view()

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    with nogil:
        c_result = move(
            cpp_hash_partition(
                c_source_view,
                c_columns_to_hash,
                c_num_partitions
            )
        )

    # Note that the offsets (`c_result.second`) may be empty when
    # the original table (`source_table`) is empty. We need to
    # return a list of zeros in this case.
    return (
        Table.from_unique_ptr(
            move(c_result.first),
            column_names=source_table._column_names,
            index_names=source_table._index_names if(
                keep_index is True)
            else None

        ),
        list(c_result.second) if c_result.second.size()
        else [0] * num_partitions
    )


def hash(Table source_table, object initial_hash_values=None):
    cdef vector[uint32_t] c_initial_hash = initial_hash_values or []
    cdef table_view c_source_view = source_table.data_view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_hash(
                c_source_view,
                libcudf_types.hash_id.HASH_MURMUR3,
                c_initial_hash
            )
        )

    return Column.from_unique_ptr(move(c_result))
