# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow.lib cimport (
    CScalar as pa_CScalar,
    CTable as pa_CTable,
    Scalar as pa_Scalar,
    Table as pa_Table,
    pyarrow_unwrap_scalar,
    pyarrow_unwrap_table,
    pyarrow_wrap_scalar,
    pyarrow_wrap_table,
)

from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table

from .scalar cimport Scalar
from .table cimport Table


cdef class ColumnMetadata:
    def __init__(self, name):
        self.name = name
        self.children_meta = []

    cdef column_metadata to_c_metadata(self):
        """Convert to C++ column_metadata.

        Since this class is mutable and cheap, it is easier to create the C++
        object on the fly rather than have it directly backing the storage for
        the Cython class.
        """
        cdef column_metadata c_metadata
        cdef ColumnMetadata child_meta
        c_metadata.name = self.name.encode()
        for child_meta in self.children_meta:
            c_metadata.children_meta.push_back(child_meta.to_c_metadata())
        return c_metadata


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


cpdef pa_Table to_arrow(Table tbl, list metadata):
    cdef shared_ptr[pa_CTable] c_result
    cdef vector[column_metadata] c_metadata
    cdef ColumnMetadata meta
    for meta in metadata:
        c_metadata.push_back(meta.to_c_metadata())

    with nogil:
        c_result = move(cpp_to_arrow(tbl.view(), c_metadata))

    return pyarrow_wrap_table(c_result)


cpdef pa_Scalar to_arrow_scalar(Scalar slr, ColumnMetadata metadata):
    cdef shared_ptr[pa_CScalar] c_result
    cdef column_metadata c_metadata = metadata.to_c_metadata()

    with nogil:
        c_result = move(cpp_to_arrow(dereference(slr.c_obj.get()), c_metadata))

    return pyarrow_wrap_scalar(c_result)
