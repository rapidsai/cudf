# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.types as libcudf_types
from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.hash cimport (
    md5,
    murmurhash3_x86_32,
    sha1,
    sha224,
    sha256,
    sha384,
    sha512,
    xxhash_64,
)
from cudf._lib.pylibcudf.libcudf.partitioning cimport (
    hash_partition as cpp_hash_partition,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
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
    cdef table_view c_source_view = table_view_from_columns(source_columns)
    cdef unique_ptr[column] c_result
    if method == "murmur3":
        with nogil:
            c_result = move(murmurhash3_x86_32(c_source_view, seed))
    elif method == "md5":
        with nogil:
            c_result = move(md5(c_source_view))
    elif method == "sha1":
        with nogil:
            c_result = move(sha1(c_source_view))
    elif method == "sha224":
        with nogil:
            c_result = move(sha224(c_source_view))
    elif method == "sha256":
        with nogil:
            c_result = move(sha256(c_source_view))
    elif method == "sha384":
        with nogil:
            c_result = move(sha384(c_source_view))
    elif method == "sha512":
        with nogil:
            c_result = move(sha512(c_source_view))
    elif method == "xxhash64":
        with nogil:
            c_result = move(xxhash_64(c_source_view, seed))
    else:
        raise ValueError(f"Unsupported hash function: {method}")
    return Column.from_unique_ptr(move(c_result))
