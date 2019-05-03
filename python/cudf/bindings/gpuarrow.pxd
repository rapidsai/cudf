# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    ctypedef struct _OpaqueIpcParser:
        pass
    ctypedef struct  gdf_ipc_parser_type:
        pass

    cdef gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length) except +
    cdef void gdf_ipc_parser_open_recordbatches(
        gdf_ipc_parser_type *handle,
        const uint8_t *recordbatches,
        size_t length
    ) except +

    cdef void gdf_ipc_parser_close(gdf_ipc_parser_type *handle) except +
    cdef int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle) except +
    cdef const char* gdf_ipc_parser_to_json(gdf_ipc_parser_type *handle) except +
    cdef const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle) except +
    cdef const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle) except +
    cdef int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle) except +

    cdef const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle) except +
    cdef const char *gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle) except +
