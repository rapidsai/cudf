from cudf._lib.cpp.lists.count_elements cimport count_elements as cpp_count_elements
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column

from cudf._lib.column cimport Column


from cudf.core.dtypes import ListDtype

def count_elements(Column col):
    if not isinstance(col.dtype, ListDtype):
        raise TypeError("col is not a list column.")
    
    cdef lists_column_view list_view = lists_column_view(col.view())
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_count_elements(list_view)
    
    result = Column.from_unique_ptr(c_result)
    return result
