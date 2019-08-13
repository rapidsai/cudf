# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libc.stdint cimport (  # noqa: E211
    uint8_t,
    uint32_t,
    int64_t,
    int32_t,
    int16_t,
    int8_t,
    uintptr_t
)
from libcpp.vector cimport vector

# Utility functions to build gdf_columns, gdf_context and error handling

cpdef get_ctype_ptr(obj)
cpdef get_column_data_ptr(obj)
cpdef get_column_valid_ptr(obj)

cpdef gdf_time_unit np_dtype_to_gdf_time_unit(dtype)
cpdef gdf_time_unit_to_np_dtype(gdf_time_unit time_unit)

cdef np_dtype_from_gdf_column(gdf_column* col)


cdef get_scalar_value(gdf_scalar scalar, dtype)

cdef gdf_column* column_view_from_column(col, col_name=*) except? NULL
cdef gdf_column* column_view_from_NDArrays(
    size,
    data,
    mask,
    dtype,
    null_count
) except? NULL
cdef gdf_scalar* gdf_scalar_from_scalar(val, dtype=*) except? NULL
cdef gdf_column_to_column(gdf_column* c_col, int_col_name=*)
cdef gdf_column_to_column_mem(gdf_column* input_col)
cdef update_nvstrings_col(col, uintptr_t category_ptr)
cdef gdf_column* column_view_from_string_column(col, col_name=*) except? NULL
cdef gdf_column** cols_view_from_cols(cols)
cdef free_table(cudf_table* table0, gdf_column** cols=*)
cdef free_column(gdf_column* c_col)

cdef gdf_context* create_context_view(
    flag_sorted,
    method,
    flag_distinct,
    flag_sort_result,
    flag_sort_inplace,
    null_sort_behavior
)

cpdef uintptr_t column_view_pointer(col)

cpdef check_gdf_error(errcode)

# Import cudf.h header to import all functions
# First version of bindings has no changes to the cudf.h header, so this file
# mirrors the structure in cpp/include

cdef extern from "cudf/cudf.h" nogil:

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

    ctypedef enum gdf_null_sort_behavior:
        GDF_NULL_AS_LARGEST = 0,
        GDF_NULL_AS_SMALLEST,

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
        GDF_NUMBA_GENERIC_AGG_OPS,
        GDF_CUDA_GENERIC_AGG_OPS,
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
        signed char   si08
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

    cdef gdf_error gdf_column_view(
        gdf_column *column,
        void *data,
        gdf_valid_type *valid,
        gdf_size_type size,
        gdf_dtype dtype
    ) except +

    # version with name parameter
    cdef gdf_error gdf_column_view_augmented(
        gdf_column *column,
        void *data,
        gdf_valid_type *valid,
        gdf_size_type size,
        gdf_dtype dtype,
        gdf_size_type null_count,
        gdf_dtype_extra_info extra_info,
        const char* name) except +

    # version without name parameter
    cdef gdf_error gdf_column_view_augmented(
        gdf_column *column,
        void *data,
        gdf_valid_type *valid,
        gdf_size_type size,
        gdf_dtype dtype,
        gdf_size_type null_count,
        gdf_dtype_extra_info extra_info
    ) except +

    cdef gdf_error gdf_column_free(gdf_column *column) except +

    cdef gdf_error gdf_context_view(
        gdf_context *context,
        int flag_sorted,
        gdf_method flag_method,
        int flag_distinct,
        int flag_sort_result,
        int flag_sort_inplace,
        gdf_null_sort_behavior flag_null_sort_behavior
    ) except +

    cdef const char * gdf_error_get_name(gdf_error errcode) except +

    cdef int gdf_cuda_last_error() except +
    cdef const char * gdf_cuda_error_string(int cuda_error) except +
    cdef const char * gdf_cuda_error_name(int cuda_error) except +

    cdef gdf_error gdf_validity_and(
        gdf_column *lhs,
        gdf_column *rhs,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_group_by_sum(
        int ncols,
        gdf_column** cols,
        gdf_column* col_agg,
        gdf_column* out_col_indices,
        gdf_column** out_col_values,
        gdf_column* out_col_agg,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_group_by_min(
        int ncols,
        gdf_column** cols,
        gdf_column* col_agg,
        gdf_column* out_col_indices,
        gdf_column** out_col_values,
        gdf_column* out_col_agg,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_group_by_max(
        int ncols,
        gdf_column** cols,
        gdf_column* col_agg,
        gdf_column* out_col_indices,
        gdf_column** out_col_values,
        gdf_column* out_col_agg,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_group_by_avg(
        int ncols,
        gdf_column** cols,
        gdf_column* col_agg,
        gdf_column* out_col_indices,
        gdf_column** out_col_values,
        gdf_column* out_col_agg,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_group_by_count(
        int ncols,
        gdf_column** cols,
        gdf_column* col_agg,
        gdf_column* out_col_indices,
        gdf_column** out_col_values,
        gdf_column* out_col_agg,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_quantile_exact(
        gdf_column* col_in,
        gdf_quantile_method prec,
        double q,
        gdf_scalar* result,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_quantile_approx(
        gdf_column* col_in,
        double q,
        gdf_scalar* result,
        gdf_context* ctxt
    ) except +

    cdef gdf_error gdf_digitize(
        gdf_column* col,
        gdf_column* bins,
        bool right,
        gdf_index_type* out_indices
    ) except +

    cdef gdf_error gdf_nvtx_range_push(
        const char* const name,
        gdf_color color
    ) except +

    cdef gdf_error gdf_nvtx_range_push_hex(
        const char* const name,
        unsigned int color
    ) except +

    cdef gdf_error gdf_nvtx_range_pop() except +


cdef extern from "cudf/legacy/bitmask.hpp" nogil:

    cdef gdf_error gdf_count_nonzero_mask(
        gdf_valid_type* masks,
        int num_rows,
        int* count
    ) except +


cdef extern from "cudf/legacy/table.hpp" namespace "cudf" nogil:

    cdef cppclass cudf_table "cudf::table":

        cudf_table(
            gdf_column* cols[],
            gdf_size_type num_cols
        ) except +

        cudf_table(const vector[gdf_column*] cols) except +

        cudf_table() except +

        gdf_column** begin() except +

        gdf_column** end() except +

        gdf_column* get_column(gdf_index_type index) except +

        gdf_size_type num_columns() except +

        gdf_size_type num_rows() except +

# Todo? add const overloads
#        const gdf_column* const* begin() const except +
#        gdf_column const* const* end() const
#        gdf_column const* get_column(gdf_index_type index) const except +

cpdef gdf_dtype gdf_dtype_from_value(col, dtype=*) except? GDF_invalid
