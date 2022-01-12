# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.hash cimport hash as cpp_hash
from cudf._lib.cpp.partitioning cimport hash_partition as cpp_hash_partition
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table


def hash_partition(source_table, object columns_to_hash,
                   int num_partitions, bool keep_index=True):
    cdef vector[libcudf_types.size_type] c_columns_to_hash = columns_to_hash
    cdef int c_num_partitions = num_partitions
    cdef table_view c_source_view = table_view_from_table(
        source_table, not keep_index
    )

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
        *data_from_unique_ptr(
            move(c_result.first),
            column_names=source_table._column_names,
            index_names=(
                source_table._index_names
                if keep_index is True
                else None
            )

        ),
        list(c_result.second) if c_result.second.size()
        else [0] * num_partitions
    )


def hash(source_table, str method, int seed=0):
    cdef table_view c_source_view = table_view_from_table(
        source_table, ignore_index=True)
    cdef unique_ptr[column] c_result
    cdef libcudf_types.hash_id c_hash_function
    if method == "murmur3":
        c_hash_function = libcudf_types.hash_id.HASH_MURMUR3
    elif method == "md5":
        c_hash_function = libcudf_types.hash_id.HASH_MD5
    else:
        raise ValueError(f"Unsupported hash function: {method}")
    with nogil:
        c_result = move(
            cpp_hash(
                c_source_view,
                c_hash_function,
                seed
            )
        )

    return Column.from_unique_ptr(move(c_result))
