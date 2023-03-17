# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.minhash cimport minhash as cpp_minhash
from cudf._lib.cpp.types cimport size_type


@acquire_spill_lock()
def minhash(Column strings, int width, int seed=0):

    cdef column_view c_strings = strings.view()
    cdef size_type c_width = width
    cdef uint32_t c_seed = seed
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_minhash(
                c_strings,
                c_width,
                c_seed
            )
        )

    return Column.from_unique_ptr(move(c_result))
