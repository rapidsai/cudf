# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib import pylibcudf

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.column cimport Column
from cudf._lib.cpp.partitioning cimport hash_partition as cpp_hash_partition
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns


@acquire_spill_lock()
def hash_partition(list source_columns, object columns_to_hash,
                   int num_partitions):
    cdef vector[libcudf_types.size_type] c_columns_to_hash = columns_to_hash
    cdef int c_num_partitions = num_partitions
    cdef table_view c_source_view = table_view_from_columns(source_columns)

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    with nogil:
        c_result = move(
            cpp_hash_partition(
                c_source_view,
                c_columns_to_hash,
                c_num_partitions
            )
        )

    return (
        columns_from_unique_ptr(move(c_result.first)),
        list(c_result.second)
    )


@acquire_spill_lock()
def hash(list source_columns, str method, int seed=0):
    ctbl = pylibcudf.Table([c.to_pylibcudf(mode="read") for c in source_columns])
    return Column.from_pylibcudf(pylibcudf.hashing.hash(ctbl, method, seed))
