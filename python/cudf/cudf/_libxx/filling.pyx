import pandas as pd

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table


def repeat(Table input, Column count):
    assert pd.api.types.is_integer_dtype(count.dtype)
    cdef unique_ptr[table] c_result = (
        cpp_repeat(
            input.view(),
            count.view(),
            True
        )
    )
    return Table.from_unique_ptr(
        move(c_result),
        column_names=input._column_names,
        index_names=input._index._column_names
    )
