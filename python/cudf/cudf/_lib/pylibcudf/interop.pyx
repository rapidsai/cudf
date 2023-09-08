# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from pyarrow.lib cimport (
    CScalar as pa_CScalar,
    CTable as pa_CTable,
    Scalar as pa_Scalar,
    Table as pa_Table,
    pyarrow_unwrap_scalar,
    pyarrow_unwrap_table,
)

from cudf._lib.cpp.interop cimport from_arrow as cpp_from_arrow
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table

from .scalar cimport Scalar
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


cpdef Scalar from_arrow_scalar(
    pa_Scalar pyarrow_scalar,
):
    cdef shared_ptr[pa_CScalar] cscalar = (
        pyarrow_unwrap_scalar(pyarrow_scalar)
    )
    cdef unique_ptr[scalar] c_result

    with nogil:
        c_result = move(cpp_from_arrow(cscalar.get()[0]))

    return Scalar.from_libcudf(move(c_result))
