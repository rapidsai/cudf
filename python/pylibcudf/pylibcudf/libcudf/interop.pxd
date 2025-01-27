# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +libcudf_exception_handler


# The Arrow structs are not namespaced.
cdef extern from "cudf/interop.hpp" nogil:
    cdef struct ArrowSchema:
        void (*release)(ArrowSchema*) noexcept

    cdef struct ArrowArray:
        void (*release)(ArrowArray*) noexcept

    cdef struct ArrowArrayStream:
        void (*release)(ArrowArrayStream*) noexcept

    cdef struct ArrowDeviceArray:
        ArrowArray array


cdef extern from "cudf/interop.hpp" namespace "cudf" \
        nogil:
    cdef unique_ptr[table] from_dlpack(
        const DLManagedTensor* managed_tensor
    ) except +libcudf_exception_handler

    DLManagedTensor* to_dlpack(
        const table_view& input
    ) except +libcudf_exception_handler

    cdef cppclass column_metadata:
        column_metadata() except +libcudf_exception_handler
        column_metadata(string name_) except +libcudf_exception_handler
        string name
        vector[column_metadata] children_meta

    cdef unique_ptr[table] from_arrow_stream(
        ArrowArrayStream* input
    ) except +libcudf_exception_handler
    cdef unique_ptr[column] from_arrow_column(
        const ArrowSchema* schema,
        const ArrowArray* input
    ) except +libcudf_exception_handler


cdef extern from *:
    # Rather than exporting the underlying functions directly to Cython, we expose
    # these wrappers that handle the release to avoid needing to teach Cython how
    # to handle unique_ptrs with custom deleters that aren't default constructible.
    # This will go away once we introduce cudf::arrow_column (need a
    # cudf::arrow_schema as well), see
    # https://github.com/rapidsai/cudf/issues/16104.
    """
    #include <nanoarrow/nanoarrow.h>
    #include <nanoarrow/nanoarrow_device.h>

    ArrowSchema* to_arrow_schema_raw(
      cudf::table_view const& input,
      cudf::host_span<cudf::column_metadata const> metadata) {
      return to_arrow_schema(input, metadata).release();
    }

    ArrowArray* to_arrow_host_raw(
      cudf::table_view const& tbl,
      rmm::cuda_stream_view stream       = cudf::get_default_stream(),
      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
      // Assumes the sync event is null and the data is already on the host.
      ArrowArray *arr = new ArrowArray();
      auto device_arr = cudf::to_arrow_host(tbl, stream, mr);
      ArrowArrayMove(&device_arr->array, arr);
      return arr;
    }
    """
    cdef ArrowSchema *to_arrow_schema_raw(
        const table_view& tbl,
        const vector[column_metadata]& metadata,
    ) except +libcudf_exception_handler nogil
    cdef ArrowArray* to_arrow_host_raw(
        const table_view& tbl
    ) except +libcudf_exception_handler nogil
