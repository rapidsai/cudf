from cudf._lib.cpp.interop cimport gather_metadata
from cudf._lib.cpp.interop cimport column_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport table_view_from_columns
from cudf._lib.cpp.ipc cimport export_ipc as cpp_export_icp
from libcpp.vector cimport vector
from libcpp.memory cimport make_shared, shared_ptr
from pyarrow._cuda cimport CCudaContext, Context

def export_ipc(list source_columns, object metadata, object ctx):
    cdef vector[column_metadata] cpp_metadata = gather_metadata(metadata)
    cdef table_view input_table_view = table_view_from_columns(source_columns)
    cdef shared_ptr[CCudaContext] cctx = (<Context>ctx).context

    cpp_export_icp(cctx, input_table_view, cpp_metadata)
    pass


def import_ipc(bytearray message):
    pass
