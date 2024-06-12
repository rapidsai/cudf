# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.transpose cimport transpose as cpp_transpose
from cudf._lib.utils cimport columns_from_table_view, table_view_from_columns


def transpose(list source_columns):
    """Transpose m n-row columns into n m-row columns
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef table_view c_input = table_view_from_columns(source_columns)

    with nogil:
        c_result = move(cpp_transpose(c_input))

    # Notice, the data pointer of `result_owner` has been exposed
    # through `c_result.second` at this point.
    result_owner = Column.from_unique_ptr(
        move(c_result.first), data_ptr_exposed=True
    )
    return columns_from_table_view(
        c_result.second,
        owners=[result_owner] * c_result.second.num_columns()
    )
