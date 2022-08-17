# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf._lib.cpp.interop cimport gather_metadata, column_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport table_view_from_columns, columns_from_table_view

from cudf._lib.cpp.ipc cimport export_ipc as cpp_export_ipc
from cudf._lib.cpp.ipc cimport import_ipc as cpp_import_ipc
from cudf._lib.cpp.ipc cimport imported_column

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move

from pyarrow.lib cimport CBuffer, pyarrow_wrap_buffer, pyarrow_unwrap_buffer
from cython.operator cimport dereference

def export_ipc(list source_columns, object metadata):
    cdef:
        vector[column_metadata] cpp_metadata = gather_metadata(metadata)
        table_view input_table_view = table_view_from_columns(source_columns)
        shared_ptr[CBuffer] cbuf = cpp_export_ipc(input_table_view, cpp_metadata)
    return pyarrow_wrap_buffer(cbuf)


cdef class ImportedColumn:
    cdef shared_ptr[imported_column] _handle

    @staticmethod
    cdef ImportedColumn from_shared_ptr(shared_ptr[imported_column] c_col):
        col = ImportedColumn()
        col._handle = c_col


def import_ipc(object message):
    cdef:
        shared_ptr[CBuffer] cbuf = pyarrow_unwrap_buffer(message)
        pair[table_view, vector[shared_ptr[imported_column]]] result = move(
            cpp_import_ipc(cbuf)
        )

    owners = [ImportedColumn.from_shared_ptr(n) for n in result.second]
    names = [dereference(n).name.decode() for n in result.second]

    columns = columns_from_table_view(move(result.first), owners)
    data = {n: columns[i] for i, n in enumerate(names)}
    return data
