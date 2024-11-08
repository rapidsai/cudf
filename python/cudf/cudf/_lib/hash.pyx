# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.table cimport Table

from cudf._lib.column cimport Column


@acquire_spill_lock()
def hash_partition(list source_columns, list columns_to_hash,
                   int num_partitions):
    plc_table, offsets = plc.partitioning.hash_partition(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_columns]),
        columns_to_hash,
        num_partitions
    )
    return [Column.from_pylibcudf(col) for col in plc_table.columns()], offsets


@acquire_spill_lock()
def hash(list source_columns, str method, int seed=0):
    cdef Table ctbl = Table(
        [c.to_pylibcudf(mode="read") for c in source_columns]
    )
    if method == "murmur3":
        return Column.from_pylibcudf(plc.hashing.murmurhash3_x86_32(ctbl, seed))
    elif method == "xxhash64":
        return Column.from_pylibcudf(plc.hashing.xxhash_64(ctbl, seed))
    elif method == "md5":
        return Column.from_pylibcudf(plc.hashing.md5(ctbl))
    elif method == "sha1":
        return Column.from_pylibcudf(plc.hashing.sha1(ctbl))
    elif method == "sha224":
        return Column.from_pylibcudf(plc.hashing.sha224(ctbl))
    elif method == "sha256":
        return Column.from_pylibcudf(plc.hashing.sha256(ctbl))
    elif method == "sha384":
        return Column.from_pylibcudf(plc.hashing.sha384(ctbl))
    elif method == "sha512":
        return Column.from_pylibcudf(plc.hashing.sha512(ctbl))
    else:
        raise ValueError(
            f"Unsupported hashing algorithm {method}."
        )
