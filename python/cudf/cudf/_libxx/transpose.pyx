# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
from cudf.core.column.column import build_categorical_column
from cudf.utils.dtypes import is_categorical_dtype

from cudf._libxx.lib cimport *
from cudf._libxx.table cimport Table
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

    if dtype.kind in 'OU':
        raise NotImplementedError('Cannot transpose string columns')
    elif not is_categorical_dtype(dtype):
        if any(c.dtype != dtype for c in source._columns):
            raise ValueError('Columns must all have the same dtype')
    else:
        if any(not is_categorical_dtype(c.dtype) for c in source._columns):
            raise ValueError('Columns must all have the same dtype')
        cats = list(c.cat().categories for c in source._columns)
        cats = cudf.Series(cudf.concat(cats)).drop_duplicates()._column
        source = Table(index=source._index, data=[
            (name, col.cat()._set_categories(cats, is_unique=True).codes)
            for name, col in source._data.items()
        ])

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
            (name, build_categorical_column(
                codes=col,
                mask=col.mask,
                categories=cats
            ))
            for name, col in result._data.items()
        ])

    return result
