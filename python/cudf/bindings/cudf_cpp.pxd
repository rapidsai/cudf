# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.dlpack cimport DLManagedTensor

from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, int64_t, int32_t, int16_t, int8_t, uintptr_t

# Utility functions to build gdf_columns, gdf_context and error handling

cpdef get_ctype_ptr(obj)
cpdef get_column_data_ptr(obj)
cpdef get_column_valid_ptr(obj)

cdef gdf_dtype get_dtype(dtype)

cdef get_scalar_value(gdf_scalar scalar)

cdef gdf_column* column_view_from_column(col)
cdef gdf_column* column_view_from_NDArrays(size, data, mask, dtype, null_count)
cdef gdf_column_to_column_mem(gdf_column* input_col)
cdef update_nvstrings_col(col, uintptr_t category_ptr)

cdef gdf_context* create_context_view(flag_sorted, method, flag_distinct,
                                      flag_sort_result, flag_sort_inplace)

cpdef check_gdf_error(errcode)

# Import cudf.h header to import all functions
# First version of bindings has no changes to the cudf.h header, so this file
# mirrors the structure in cpp/include

cdef extern from "cudf.h" nogil:

    ctypedef int           gdf_size_type
    ctypedef gdf_size_type gdf_index_type
    ctypedef unsigned char gdf_valid_type
    ctypedef char          gdf_bool8
    ctypedef long          gdf_date64
    ctypedef int           gdf_date32
    ctypedef long          gdf_timestamp
    ctypedef int           gdf_category
    ctypedef int           gdf_nvstring_category

    ctypedef enum gdf_dtype:
        GDF_invalid=0,
        GDF_INT8,
        GDF_INT16,
        GDF_INT32,
        GDF_INT64,
        GDF_FLOAT32,
        GDF_FLOAT64,
        GDF_BOOL8,
        GDF_DATE32,
        GDF_DATE64,
        GDF_TIMESTAMP,
        GDF_CATEGORY,
        GDF_STRING,
        GDF_STRING_CATEGORY,
        N_GDF_TYPES,

    ctypedef enum gdf_error:
        GDF_SUCCESS=0,
        GDF_CUDA_ERROR,
        GDF_UNSUPPORTED_DTYPE,
        GDF_COLUMN_SIZE_MISMATCH,
        GDF_COLUMN_SIZE_TOO_BIG,
        GDF_DATASET_EMPTY,
        GDF_VALIDITY_MISSING,
        GDF_VALIDITY_UNSUPPORTED,
        GDF_INVALID_API_CALL,
        GDF_JOIN_DTYPE_MISMATCH,
        GDF_JOIN_TOO_MANY_COLUMNS,
        GDF_DTYPE_MISMATCH,
        GDF_UNSUPPORTED_METHOD,
        GDF_INVALID_AGGREGATOR,
        GDF_INVALID_HASH_FUNCTION,
        GDF_PARTITION_DTYPE_MISMATCH,
        GDF_HASH_TABLE_INSERT_FAILURE,
        GDF_UNSUPPORTED_JOIN_TYPE,
        GDF_C_ERROR,
        GDF_FILE_ERROR,
        GDF_MEMORYMANAGER_ERROR,
        GDF_UNDEFINED_NVTX_COLOR,
        GDF_NULL_NVTX_NAME,
        GDF_NOTIMPLEMENTED_ERROR,
        N_GDF_ERRORS

    ctypedef enum gdf_time_unit:
        TIME_UNIT_NONE=0
        TIME_UNIT_s,
        TIME_UNIT_ms,
        TIME_UNIT_us,
        TIME_UNIT_ns

    ctypedef struct gdf_dtype_extra_info:
        gdf_time_unit time_unit
        void *category

    ctypedef struct gdf_column:
        void *data
        gdf_valid_type *valid
        gdf_size_type size
        gdf_dtype dtype
        gdf_size_type null_count
        gdf_dtype_extra_info dtype_info
        char *col_name

    ctypedef enum gdf_method:
      GDF_SORT = 0,
      GDF_HASH,
      N_GDF_METHODS,

    ctypedef enum gdf_quantile_method:
      GDF_QUANT_LINEAR =0,
      GDF_QUANT_LOWER,
      GDF_QUANT_HIGHER,
      GDF_QUANT_MIDPOINT,
      GDF_QUANT_NEAREST,
      N_GDF_QUANT_METHODS,

    ctypedef enum gdf_agg_op:
      GDF_SUM = 0,
      GDF_MIN,
      GDF_MAX,
      GDF_AVG,
      GDF_COUNT,
      GDF_COUNT_DISTINCT,
      N_GDF_AGG_OPS,

    ctypedef enum gdf_color:
      GDF_GREEN = 0,
      GDF_BLUE,
      GDF_YELLOW,
      GDF_PURPLE,
      GDF_CYAN,
      GDF_RED,
      GDF_WHITE,
      GDF_DARK_GREEN,
      GDF_ORANGE,
      GDF_NUM_COLORS,

    ctypedef struct gdf_context:
      int flag_sorted
      gdf_method flag_method
      int flag_distinct
      int flag_sort_result
      int flag_sort_inplace

    ctypedef struct _OpaqueIpcParser:
        pass
    ctypedef struct  gdf_ipc_parser_type:
        pass

    

    ctypedef enum window_function_type:
        GDF_WINDOW_RANGE,
        GDF_WINDOW_ROW

    ctypedef enum window_reduction_type:
        GDF_WINDOW_AVG,
        GDF_WINDOW_SUM,
        GDF_WINDOW_MAX,
        GDF_WINDOW_MIN,
        GDF_WINDOW_COUNT,
        GDF_WINDOW_STDDEV,
        GDF_WINDOW_VA

    ctypedef union gdf_data:
        char          si08
        short         si16
        int           si32
        long          si64
        float         fp32
        double        fp64
        char          b08
        gdf_date32    dt32
        gdf_date64    dt64
        gdf_timestamp tmst

    ctypedef struct gdf_scalar:
        gdf_data  data
        gdf_dtype dtype
        bool      is_valid

    cdef gdf_size_type gdf_column_sizeof() except +

    cdef gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                                   gdf_size_type size, gdf_dtype dtype) except +

    cdef gdf_error gdf_column_view_augmented(gdf_column *column,
                                             void *data,
                                             gdf_valid_type *valid,
                                             gdf_size_type size,
                                             gdf_dtype dtype,
                                             gdf_size_type null_count,
                                             gdf_dtype_extra_info extra_info) except +

    cdef gdf_error gdf_column_free(gdf_column *column) except +

    cdef gdf_error gdf_context_view(gdf_context *context,
                                    int flag_sorted,
                                    gdf_method flag_method,
                                    int flag_distinct,
                                    int flag_sort_result,
                                    int flag_sort_inplace) except +

    cdef const char * gdf_error_get_name(gdf_error errcode) except +

    cdef int gdf_cuda_last_error() except +
    cdef const char * gdf_cuda_error_string(int cuda_error) except +
    cdef const char * gdf_cuda_error_name(int cuda_error) except +

    cdef gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length) except +
    cdef void gdf_ipc_parser_open_recordbatches(gdf_ipc_parser_type *handle,
                                           const uint8_t *recordbatches,
                                           size_t length) except +

    cdef void gdf_ipc_parser_close(gdf_ipc_parser_type *handle) except +
    cdef int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle) except +
    cdef const char* gdf_ipc_parser_to_json(gdf_ipc_parser_type *handle) except +
    cdef const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle) except +
    cdef const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle) except +
    cdef int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle) except +

    cdef const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle) except +
    cdef const char *gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle) except +

    cdef gdf_error gdf_cast(gdf_column *input, gdf_column *output) except +

    cdef gdf_error gdf_validity_and(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    
    cdef gdf_error gdf_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output) except +

    cdef gdf_size_type gdf_dtype_size(gdf_dtype dtype) except +

    cdef gdf_error get_column_byte_width(gdf_column * col, int * width) except +

    cdef gdf_error gdf_filter(size_t nrows,
                 gdf_column* cols,
                 size_t ncols,
                 void** d_cols,
                 int* d_types,
                 void** d_vals,
                 size_t* d_indx,
                 size_t* new_sz) except +

    cdef gdf_error gdf_group_by_sum(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt) except +

    cdef gdf_error gdf_group_by_min(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt) except +


    cdef gdf_error gdf_group_by_max(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt) except +


    cdef gdf_error gdf_group_by_avg(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt) except +

    cdef gdf_error gdf_group_by_count(int ncols,
                                 gdf_column** cols,
                                 gdf_column* col_agg,
                                 gdf_column* out_col_indices,
                                 gdf_column** out_col_values,

                                 gdf_column* out_col_agg,
                                 gdf_context* ctxt) except +


    cdef gdf_error gdf_quantile_exact(gdf_column*       col_in,
                                    gdf_quantile_method prec,
                                    double              q,
                                    gdf_scalar*         result,
                                    gdf_context*        ctxt) except +


    cdef gdf_error gdf_quantile_approx(  gdf_column*  col_in,
                                    double       q,
                                    gdf_scalar*  result,
                                    gdf_context* ctxt) except +


    cdef gdf_error gdf_find_and_replace_all(gdf_column*       col,
                                   gdf_column* old_values,
                                   gdf_column* new_values) except +


    cdef gdf_error gdf_replace_nulls(gdf_column* col_out,
                                     const gdf_column* col_in) except +


    cdef gdf_error gdf_digitize(gdf_column* col,
                                gdf_column* bins,
                                bool right,
                                gdf_index_type* out_indices) except +

    cdef gdf_error gdf_from_dlpack(gdf_column** columns,
                                   gdf_size_type *num_columns,
                                   const DLManagedTensor * tensor) except +

    cdef gdf_error gdf_to_dlpack(DLManagedTensor *tensor,
                                 const gdf_column ** columns,
                                 gdf_size_type num_columns) except +

    cdef gdf_error gdf_nvtx_range_push(const char * const name, gdf_color color ) except +

    cdef gdf_error gdf_nvtx_range_push_hex(const char * const name, unsigned int color ) except +

    cdef gdf_error gdf_nvtx_range_pop() except +


cdef extern from "bitmask.hpp" nogil:

    cdef gdf_error gdf_count_nonzero_mask(gdf_valid_type * masks, int num_rows, int * count) except +
