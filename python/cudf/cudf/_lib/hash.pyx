# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector


import pylibcudf
cimport pylibcudf.libcudf.types as libcudf_types
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.hash cimport (
    md5,
    murmurhash3_x86_32,
    sha1,
    sha224,
    sha256,
    sha384,
    sha512,
    xxhash_64,
)
from pylibcudf.libcudf.partitioning cimport (
    hash_partition as cpp_hash_partition,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view

from cudf._lib.column cimport Column
from pylibcudf.table cimport Table
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
    cdef Table ctbl = Table(
        [c.to_pylibcudf(mode="read") for c in source_columns]
    )
    if method == "murmur3":
        return Column.from_pylibcudf(pylibcudf.hashing.murmurhash3_x86_32(ctbl, seed))
    elif method == "xxhash64":
        return Column.from_pylibcudf(pylibcudf.hashing.xxhash_64(ctbl, seed))
    elif method == "md5":
        return Column.from_pylibcudf(pylibcudf.hashing.md5(ctbl))
    elif method == "sha1":
        return Column.from_pylibcudf(pylibcudf.hashing.sha1(ctbl))
    elif method == "sha224":
        return Column.from_pylibcudf(pylibcudf.hashing.sha224(ctbl))
    elif method == "sha256":
        return Column.from_pylibcudf(pylibcudf.hashing.sha256(ctbl))
    elif method == "sha384":
        return Column.from_pylibcudf(pylibcudf.hashing.sha384(ctbl))
    elif method == "sha512":
        return Column.from_pylibcudf(pylibcudf.hashing.sha512(ctbl))
    else:
        raise ValueError(
            f"Unsupported hashing algorithm {method}."
        )
