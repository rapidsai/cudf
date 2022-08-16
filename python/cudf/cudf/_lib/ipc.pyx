from cudf._lib.cpp.interop cimport gather_metadata
from cudf._lib.cpp.interop cimport column_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport table_view_from_columns
from libcpp.vector cimport vector

def export_ipc(list source_columns, object metadata):
    cdef vector[column_metadata] cpp_metadata = gather_metadata(metadata)
    cdef table_view input_table_view = table_view_from_columns(source_columns)
    pass


def import_ipc(bytearray message):
    pass
