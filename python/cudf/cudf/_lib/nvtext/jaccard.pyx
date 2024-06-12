# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.jaccard cimport (
    jaccard_index as cpp_jaccard_index,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type


@acquire_spill_lock()
def jaccard_index(Column input1, Column input2, int width):
    cdef column_view c_input1 = input1.view()
    cdef column_view c_input2 = input2.view()
    cdef size_type c_width = width
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_jaccard_index(
                c_input1,
                c_input2,
                c_width
            )
        )

    return Column.from_unique_ptr(move(c_result))
