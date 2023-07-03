# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.hash cimport hash_id as cpp_hash_id
from cudf._lib.cpp.nvtext.minhash cimport minhash as cpp_minhash
from cudf._lib.cpp.types cimport size_type


@acquire_spill_lock()
def minhash(Column strings, Column seeds, int width, str method):

    cdef column_view c_strings = strings.view()
    cdef size_type c_width = width
    cdef column_view c_seeds = seeds.view()
    cdef unique_ptr[column] c_result
    cdef cpp_hash_id c_hash_function
    if method == "murmur3":
        c_hash_function = cpp_hash_id.HASH_MURMUR3
    else:
        raise ValueError(f"Unsupported hash function: {method}")

    with nogil:
        c_result = move(
            cpp_minhash(
                c_strings,
                c_seeds,
                c_width,
                c_hash_function
            )
        )

    return Column.from_unique_ptr(move(c_result))
