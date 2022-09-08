# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.transpose cimport transpose as cpp_transpose
from cudf._lib.utils cimport columns_from_table_view, table_view_from_columns


def transpose(list source_columns):
    """Transpose m n-row columns into n m-row columns
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef table_view c_input = table_view_from_columns(source_columns)

    with nogil:
        c_result = move(cpp_transpose(c_input))

    result_owner = Column.from_unique_ptr(move(c_result.first))
    return columns_from_table_view(
        c_result.second,
        owners=[result_owner] * c_result.second.num_columns()
    )
