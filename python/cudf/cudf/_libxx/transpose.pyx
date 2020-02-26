# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf.utils.dtypes import is_categorical_dtype

from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view
from cudf._libxx.includes.column.column_view cimport column_view
cimport cudf._libxx.includes.transpose as cpp_transpose


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
            (name, col.cat()._set_categories(cats, is_unique=True).codes)
            for name, col in source._data.items()
        ])
    elif dtype.kind in 'OU':
        raise NotImplementedError('Cannot transpose string columns')
    elif any(c.dtype != dtype for c in source._columns):
        raise ValueError('Columns must all have the same dtype')

    cdef unique_ptr[table] c_result
    cdef table_view c_input = source.data_view()

    with nogil:
        c_result = move(cpp_transpose.transpose(c_input))

    result = Table.from_unique_ptr(
        move(c_result),
        column_names=cudf.core.index.RangeIndex(0, source._num_rows)
    )

    if cats is not None:
        result = Table(index=result._index, data=[
            (name, cudf.core.column.column.build_categorical_column(
                codes=col,
                mask=col.mask,
                categories=cats
            ))
            for name, col in result._data.items()
        ])

    return result
