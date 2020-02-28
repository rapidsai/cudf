# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from libcpp.memory cimport unique_ptr

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.column.column_view cimport column_view
cimport cudf._libxx.cpp.copying as cpp_copying


def gather(Table source_table, Column gather_map):
    assert pd.api.types.is_integer_dtype(gather_map.dtype)

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = source_table.view()
    cdef column_view gather_map_view = gather_map.view()

    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table_view,
                gather_map_view
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index._column_names
    )
