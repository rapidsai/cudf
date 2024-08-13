# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pyarrow.lib cimport CScalar, CTable

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +


# The Arrow structs are not namespaced.
cdef extern from "cudf/interop.hpp" nogil:
    cdef struct ArrowSchema:
        void (*release)(ArrowSchema*) noexcept nogil

    cdef struct ArrowArray:
        void (*release)(ArrowArray*) noexcept nogil

    cdef struct ArrowArrayStream:
        void (*release)(ArrowArrayStream*) noexcept nogil

    cdef struct ArrowDeviceArray:
        ArrowArray array


cdef extern from "cudf/interop.hpp" namespace "cudf" \
        nogil:
    cdef unique_ptr[table] from_dlpack(const DLManagedTensor* tensor
                                       ) except +

    DLManagedTensor* to_dlpack(table_view input_table
                               ) except +

    cdef unique_ptr[table] from_arrow(CTable input) except +
    cdef unique_ptr[scalar] from_arrow(CScalar input) except +

    cdef cppclass column_metadata:
        column_metadata() except +
        column_metadata(string name_) except +
        string name
        vector[column_metadata] children_meta

    cdef shared_ptr[CTable] to_arrow(
        table_view input,
        vector[column_metadata] metadata,
    ) except +

    cdef shared_ptr[CScalar] to_arrow(
        const scalar& input,
        column_metadata metadata,
    ) except +

    cdef unique_ptr[table] from_arrow_stream(ArrowArrayStream* input) except +
    cdef unique_ptr[column] from_arrow_column(
        const ArrowSchema* schema,
        const ArrowArray* input
    ) except +


cdef extern from *:
    # Rather than exporting the underlying functions directly to Cython, we expose
    # these wrappers that handle the release to avoid needing to teach Cython how
    # to handle unique_ptrs with custom deleters that aren't default constructible.
    # This will go away once we introduce cudf::arrow_column, see
    # https://github.com/rapidsai/cudf/issues/16104.
    """
    #include <nanoarrow/nanoarrow.h>

    ArrowSchema* to_arrow_schema_raw(
      cudf::table_view const& input,
      cudf::host_span<cudf::column_metadata const> metadata) {
      return to_arrow_schema(input, metadata).release();
    }

    ArrowArray* to_arrow_host_raw(
      cudf::table_view const& tbl,
      rmm::cuda_stream_view stream        = cudf::get_default_stream(),
      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
      // Assumes the sync event is null and the data is already on the host.
      ArrowArray *arr = new ArrowArray();
      auto device_arr = cudf::to_arrow_host(tbl, stream, mr);
      ArrowArrayMove(&device_arr->array, arr);
      return arr;
    }

    ArrowArray* to_arrow_host_raw(
      cudf::column_view const& col,
      rmm::cuda_stream_view stream        = cudf::get_default_stream(),
      rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
      // Assumes the sync event is null and the data is already on the host.
      ArrowArray *arr = new ArrowArray();
      auto device_arr = cudf::to_arrow_host(col, stream, mr);
      ArrowArrayMove(&device_arr->array, arr);
      return arr;
    }
    """
    cdef ArrowSchema *to_arrow_schema_raw(
        const table_view& tbl,
        const vector[column_metadata]& metadata,
    ) except + nogil
    cdef ArrowArray* to_arrow_host_raw(const table_view& tbl) except + nogil
    cdef ArrowArray* to_arrow_host_raw(const column_view& col) except + nogil
