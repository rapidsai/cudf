# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.optional cimport optional
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +libcudf_exception_handler


# The Arrow structs are not namespaced.
cdef extern from "cudf/interop.hpp" nogil:
    # https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
    cdef struct ArrowSchema:
        const char* format
        const char* name
        const char* metadata
        int64_t flags
        int64_t n_children
        ArrowSchema** children
        ArrowSchema* dictionary
        void (*release)(ArrowSchema*) noexcept
        void* private_data

    cdef struct ArrowArray:
        int64_t length
        int64_t null_count
        int64_t offset
        int64_t n_buffers
        int64_t n_children
        const void** buffers
        ArrowArray** children
        ArrowArray* dictionary
        void (*release)(ArrowArray*) noexcept
        void* private_data

    cdef struct ArrowArrayStream:
        void (*release)(ArrowArrayStream*) noexcept

    # https://arrow.apache.org/docs/format/CDeviceDataInterface.html#structure-definitions
    cdef struct ArrowDeviceArray:
        ArrowArray array


cdef extern from "cudf/interop.hpp" namespace "cudf" \
        nogil:
    cdef unique_ptr[table] from_dlpack(
        const DLManagedTensor* managed_tensor,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    DLManagedTensor* to_dlpack(
        const table_view& input,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef cppclass column_metadata:
        column_metadata() except +libcudf_exception_handler
        column_metadata(string name_) except +libcudf_exception_handler
        string name
        string timezone
        optional[int32_t] precision
        vector[column_metadata] children_meta


cdef extern from "cudf/interop.hpp" namespace "cudf::interop" \
        nogil:
    cdef cppclass arrow_column:
        arrow_column(
            ArrowSchema&& schema,
            ArrowArray&& array
        ) except +libcudf_exception_handler
        arrow_column(
            ArrowSchema&& schema,
            ArrowDeviceArray&& array
        ) except +libcudf_exception_handler
        arrow_column(
            ArrowArrayStream&& stream,
        ) except +libcudf_exception_handler
        column_view view() except +libcudf_exception_handler

    cdef cppclass arrow_table:
        arrow_table(
            ArrowArrayStream&& stream,
        ) except +libcudf_exception_handler
        arrow_table(
            ArrowSchema&& schema,
            ArrowDeviceArray&& array,
        ) except +libcudf_exception_handler
        table_view view() except +libcudf_exception_handler


cdef extern from *:
    # Rather than exporting the underlying functions directly to Cython, we expose
    # these wrappers that handle the release to avoid needing to teach Cython how
    # to handle unique_ptrs with custom deleters that aren't default constructible.
    # We cannot use cudf's owning arrow types for this because pylibcudf's
    # objects always manage data ownership independently of libcudf in order to
    # support other data sources (e.g. cupy), so we must use the view-based
    # C++ APIs and handle ownership in Python.
    """
    #include <cudf/interop.hpp>

    struct ArrowSchema {
      const char*  format;
      const char*  name;
      const char*  metadata;
      int64_t      flags;
      int64_t      n_children;
      ArrowSchema** children;
      ArrowSchema*  dictionary;
      void (*release)(ArrowSchema*);
      void*        private_data;
    };

    struct ArrowArray {
      int64_t      length;
      int64_t      null_count;
      int64_t      offset;
      int64_t      n_buffers;
      int64_t      n_children;
      const void** buffers;
      ArrowArray** children;
      ArrowArray*  dictionary;
      void (*release)(ArrowArray*);
      void*        private_data;
    };

    struct ArrowDeviceArray {
      ArrowArray   array;
      int64_t      device_id;
      int32_t      device_type;
      void*        sync_event;
      void       (*release)(ArrowDeviceArray*);
      void*        reserved[3];
    };

    inline void ArrowArrayMove(ArrowArray* src, ArrowArray* dst) {
      if (dst && dst->release) { dst->release(dst); }
      std::memcpy(dst, src, sizeof(ArrowArray));
      src->release = nullptr;
      src->private_data = nullptr;
      src->buffers = nullptr;
      src->children = nullptr;
      src->dictionary = nullptr;
    }

    inline void ArrowSchemaMove(ArrowSchema* src, ArrowSchema* dst) {
      if (dst && dst->release) { dst->release(dst); }
      std::memcpy(dst, src, sizeof(ArrowSchema));
      src->release = nullptr;
      src->private_data = nullptr;
      src->children = nullptr;
      src->dictionary = nullptr;
    }

    ArrowSchema* to_arrow_schema_raw(
      cudf::table_view const& input,
      cudf::host_span<cudf::column_metadata const> metadata) {
      return to_arrow_schema(input, metadata).release();
    }

    ArrowSchema* to_arrow_schema_raw(
      cudf::column_view const& input,
      cudf::column_metadata const& metadata) {
      std::vector<cudf::column_metadata> metadata_vec{metadata};
      cudf::table_view const& tbl = cudf::table_view({input});
      auto schema = cudf::to_arrow_schema(tbl, metadata_vec);
      ArrowSchema *array_schema = new ArrowSchema();
      ArrowSchemaMove(schema->children[0], array_schema);
      return array_schema;
    }

    void release_arrow_schema_raw(ArrowSchema *schema) {
      if (schema->release != nullptr) {
          schema->release(schema);
      }
      delete schema;
    }

    template <typename ViewType>
    ArrowArray* to_arrow_host_raw(
      ViewType const& obj,
      rmm::cuda_stream_view stream       = cudf::get_default_stream(),
      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
      ArrowArray *arr = new ArrowArray();
      auto device_arr = cudf::to_arrow_host(obj, stream, mr);
      ArrowArrayMove(&device_arr->array, arr);
      return arr;
    }

    void release_arrow_array_raw(ArrowArray *array) {
      if (array->release != nullptr) {
        array->release(array);
      }
      delete array;
    }

    void release_arrow_device_array_raw(ArrowDeviceArray *array) {
      if (array->array.release != nullptr) {
        array->array.release(&array->array);
      }
      delete array;
    }

    struct PylibcudfArrowDeviceArrayPrivateData {
       ArrowArray parent;
       PyObject* owner;
    };

    void PylibcudfArrowDeviceArrayRelease(ArrowArray* array)
    {
      auto private_data = reinterpret_cast<PylibcudfArrowDeviceArrayPrivateData*>(
        array->private_data);
      Py_DECREF(private_data->owner);
      private_data->parent.release(&private_data->parent);
      array->release = nullptr;
    }

    template <typename ViewType>
    ArrowDeviceArray* to_arrow_device_raw(
      ViewType const& obj,
      PyObject* owner,
      rmm::cuda_stream_view stream       = cudf::get_default_stream(),
      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
      auto tmp = cudf::to_arrow_device(obj, stream, mr);

      // Instead of moving the whole device array, we move the underlying ArrowArray
      // into the custom private data struct for managing its data then create a new
      // device array from scratch.
      auto private_data = new PylibcudfArrowDeviceArrayPrivateData();
      ArrowArrayMove(&tmp->array, &private_data->parent);
      private_data->owner = owner;
      Py_INCREF(owner);

      ArrowDeviceArray *arr = new ArrowDeviceArray();
      arr->device_id          = tmp->device_id;
      arr->device_type        = tmp->device_type;
      arr->sync_event         = tmp->sync_event;
      arr->array              = private_data->parent;  // shallow copy
      arr->array.private_data = private_data;
      arr->array.release      = &PylibcudfArrowDeviceArrayRelease;

      return arr;
    }
    """
    # The `to_*_raw` functions are all defined in the above extern block as wrappers
    # around libcudf functions that return unique_ptrs with non-default deleters, which
    # are nontrivial to wrap in Cython. Since we need to manage them as raw pointers in
    # Cython anyway, the inline C++ functions above are the simplest way to bridge the
    # gap from a language syntax perspective.
    #
    # The corresponding `release_*_raw` functions are needed because while the arrow
    # types are pure C structs, we allocate them with new in C++ and need to use delete
    # to free them. Unfortunately, unless we lie to Cython and tell it that these types
    # are cppclasses, Cython will not allow the usage of the del Python keyword to
    # generate the delete call, so inline C++ is again the best option.
    cdef ArrowSchema *to_arrow_schema_raw(
        const table_view& tbl,
        const vector[column_metadata]& metadata,
    ) except +libcudf_exception_handler nogil
    cdef ArrowSchema *to_arrow_schema_raw(
        const column_view& tbl,
        const column_metadata& metadata,
    ) except +libcudf_exception_handler nogil
    cdef void release_arrow_schema_raw(
        ArrowSchema *
    ) except +libcudf_exception_handler nogil
    cdef ArrowArray* to_arrow_host_raw(
        const table_view& tbl
    ) except +libcudf_exception_handler nogil
    cdef ArrowArray* to_arrow_host_raw(
        const column_view& tbl
    ) except +libcudf_exception_handler nogil
    cdef void release_arrow_array_raw(
        ArrowArray *
    ) except +libcudf_exception_handler nogil
    cdef void release_arrow_device_array_raw(
        ArrowDeviceArray *
    ) except +libcudf_exception_handler nogil
    cdef ArrowDeviceArray* to_arrow_device_raw(
        const table_view& tbl,
        object owner,
    ) except +libcudf_exception_handler nogil
    cdef ArrowDeviceArray* to_arrow_device_raw(
        const column_view& tbl,
        object owner,
    ) except +libcudf_exception_handler nogil
