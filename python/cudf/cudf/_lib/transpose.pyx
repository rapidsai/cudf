# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf.api.types import is_categorical_dtype

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.transpose cimport transpose as cpp_transpose
from cudf._lib.utils cimport data_from_table_view, table_view_from_table


def transpose(source):
    """Transpose index and columns.

    See Also
    --------
    cudf.core.DataFrame.transpose
    """

    if source._num_columns == 0:
        return source

    cats = None
    columns = source._columns
    dtype = columns[0].dtype

    if is_categorical_dtype(dtype):
        if any(not is_categorical_dtype(c.dtype) for c in columns):
            raise ValueError('Columns must all have the same dtype')
        cats = list(c.categories for c in columns)
        cats = cudf.core.column.concat_columns(cats).unique()
        source = cudf.core.frame.Frame(index=source._index, data=[
            (name, col._set_categories(cats, is_unique=True).codes)
            for name, col in source._data.items()
        ])
    elif any(c.dtype != dtype for c in columns):
        raise ValueError('Columns must all have the same dtype')

    cdef pair[unique_ptr[column], table_view] c_result
    cdef table_view c_input = table_view_from_table(
        source, ignore_index=True)

    with nogil:
        c_result = move(cpp_transpose(c_input))

    result_owner = Column.from_unique_ptr(move(c_result.first))
    data, _ = data_from_table_view(
        c_result.second,
        owner=result_owner,
        column_names=range(c_input.num_rows())
    )

    if cats is not None:
        data= [
            (name, cudf.core.column.column.build_categorical_column(
                codes=cudf.core.column.column.build_column(
                    col.base_data, dtype=col.dtype),
                mask=col.base_mask,
                size=col.size,
                categories=cats,
                offset=col.offset,
            ))
            for name, col in data.items()
        ]

    return data
