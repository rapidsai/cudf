# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf.utils.dtypes import is_categorical_dtype

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.pair cimport pair

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.transpose cimport (
    transpose as cpp_transpose
)


def transpose(Table source):
    """Transpose index and columns.

    See Also
    --------
    cudf.core.DataFrame.transpose
    """

    if source._num_columns == 0:
        return source

    cats = None
    dtype = source._columns[0].dtype

    if is_categorical_dtype(dtype):
        if any(not is_categorical_dtype(c.dtype) for c in source._columns):
            raise ValueError('Columns must all have the same dtype')
        cats = list(c.cat().categories for c in source._columns)
        cats = cudf.Series(cudf.concat(cats)).drop_duplicates()._column
        source = Table(index=source._index, data=[
            (name, col.cat()._set_categories(
                col.cat().categories, cats, is_unique=True).codes)
            for name, col in source._data.items()
        ])
    elif dtype.kind in 'OU':
        raise NotImplementedError('Cannot transpose string columns')
    elif any(c.dtype != dtype for c in source._columns):
        raise ValueError('Columns must all have the same dtype')

    cdef pair[unique_ptr[column], table_view] c_result
    cdef table_view c_input = source.data_view()

    with nogil:
        c_result = move(cpp_transpose(c_input))

    result_owner = Column.from_unique_ptr(move(c_result.first))
    result = Table.from_table_view(
        c_result.second,
        owner=result_owner,
        column_names=range(source._num_rows)
    )

    if cats is not None:
        result = Table(index=result._index, data=[
            (name, cudf.core.column.column.build_categorical_column(
                codes=cudf.core.column.column.as_column(
                    col.base_data, dtype=col.dtype),
                mask=col.base_mask,
                size=col.size,
                categories=cats,
                offset=col.offset,
            ))
            for name, col in result._data.items()
        ])

    return result
