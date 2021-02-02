# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr, shared_ptr, make_shared
from libcpp.utility cimport move

from cudf._lib.cpp.lists.count_elements cimport (
    count_elements as cpp_count_elements
)
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column

from cudf._lib.column cimport Column


from cudf.core.dtypes import ListDtype


def count_elements(Column col):
    if not isinstance(col.dtype, ListDtype):
        raise TypeError("col is not a list column.")

    # shared_ptr required because lists_column_view has no default
    # ctor
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_count_elements(list_view.get()[0]))

    result = Column.from_unique_ptr(move(c_result))
    return result
