# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.minhash cimport (
    minhash as cpp_minhash,
    minhash64 as cpp_minhash64,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type


@acquire_spill_lock()
def minhash(Column strings, Column seeds, int width):

    cdef column_view c_strings = strings.view()
    cdef size_type c_width = width
    cdef column_view c_seeds = seeds.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_minhash(
                c_strings,
                c_seeds,
                c_width
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def minhash64(Column strings, Column seeds, int width):

    cdef column_view c_strings = strings.view()
    cdef size_type c_width = width
    cdef column_view c_seeds = seeds.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_minhash64(
                c_strings,
                c_seeds,
                c_width
            )
        )

    return Column.from_unique_ptr(move(c_result))
