from cudf._lib.cpp.interop cimport gather_metadata, column_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport table_view_from_columns, data_from_unique_ptr
# columns_from_unique_ptr
from cudf._lib.cpp.ipc cimport export_ipc as cpp_export_ipc
from cudf._lib.cpp.ipc cimport import_ipc as cpp_import_ipc

from libcpp.vector cimport vector
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move

from pyarrow.lib cimport CBuffer, Buffer, pyarrow_wrap_buffer, pyarrow_unwrap_buffer
from pyarrow._cuda cimport CCudaContext, Context, CudaBuffer, CCudaBuffer

def export_ipc(object ctx, list source_columns, object metadata):
    cdef vector[column_metadata] cpp_metadata = gather_metadata(metadata)
    cdef table_view input_table_view = table_view_from_columns(source_columns)
    cdef shared_ptr[CCudaContext] cctx = (<Context>ctx).context

    cdef shared_ptr[CBuffer] cbuf = cpp_export_ipc(cctx, input_table_view, cpp_metadata)
    return pyarrow_wrap_buffer(cbuf)


def import_ipc(object ctx, object message):
    cdef shared_ptr[CCudaContext] cctx = (<Context>ctx).context
    cdef shared_ptr[CBuffer] cbuf = pyarrow_unwrap_buffer(message)
    cdef pair[unique_ptr[table], vector[string]] result = move(cpp_import_ipc(cctx, cbuf))
    columns = data_from_unique_ptr(move(result.first), result.second)
    return columns
