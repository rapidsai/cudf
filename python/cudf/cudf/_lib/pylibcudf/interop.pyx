# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from pyarrow.lib cimport (
    CTable as pa_CTable,
    Table as pa_Table,
    pyarrow_unwrap_table,
)

from cudf._lib.cpp.interop cimport from_arrow as cpp_from_arrow
from cudf._lib.cpp.table.table cimport table

from .table cimport Table


cpdef Table from_arrow(
    pa_Table pyarrow_table,
):
    cdef shared_ptr[pa_CTable] ctable = (
        pyarrow_unwrap_table(pyarrow_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(ctable.get()[0]))

    return Table.from_libcudf(move(c_result))
