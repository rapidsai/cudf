# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.dlpack cimport DLManagedTensor

from libcpp cimport bool
from libc.stdint cimport uint8_t, int64_t, int32_t, int16_t, int8_t, uintptr_t

# Utility functions to build gdf_columns, gdf_context and error handling

cpdef get_ctype_ptr(obj)
cpdef get_column_data_ptr(obj)
cpdef get_column_valid_ptr(obj)

cdef gdf_dtype get_dtype(dtype)

cdef get_scalar_value(gdf_scalar scalar)

cdef gdf_column* column_view_from_column(col)
cdef gdf_column* column_view_from_NDArrays(size, data, mask,
                                           dtype, null_count)
cdef gdf_column_to_column_mem(gdf_column* input_col)
cdef update_nvstrings_col(col, uintptr_t category_ptr)

cdef gdf_context* create_context_view(flag_sorted, method, flag_distinct,
                                      flag_sort_result, flag_sort_inplace)

cpdef check_gdf_error(errcode)

# Import cudf.h header to import all functions
# First version of bindings has no changes to the cudf.h header, so this file
# mirrors the structure in cpp/include

cdef extern from "cudf.h" nogil:

    ctypedef int gdf_size_type
    ctypedef gdf_size_type gdf_index_type
    ctypedef unsigned char gdf_valid_type
    ctypedef long    gdf_date64
    ctypedef int     gdf_date32
    ctypedef long    gdf_timestamp
    ctypedef int     gdf_category
    ctypedef int     gdf_nvstring_category

    ctypedef enum gdf_dtype:
        GDF_invalid=0,
        GDF_INT8,
        GDF_INT16,
        GDF_INT32,
        GDF_INT64,
        GDF_FLOAT32,
        GDF_FLOAT64,
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

    ctypedef enum gdf_hash_func:
        GDF_HASH_MURMUR3=0,
        GDF_HASH_IDENTITY,


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


    ctypedef struct _OpaqueRadixsortPlan:
        pass
    ctypedef struct  gdf_radixsort_plan_type:
        pass


    ctypedef struct _OpaqueSegmentedRadixsortPlan:
        pass
    ctypedef struct  gdf_segmented_radixsort_plan_type:
        pass

    ctypedef enum order_by_type:
        GDF_ORDER_ASC,
        GDF_ORDER_DESC


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


    ctypedef enum gdf_binary_operator:
        GDF_ADD,
        GDF_SUB,
        GDF_MUL,
        GDF_DIV,
        GDF_TRUE_DIV,
        GDF_FLOOR_DIV,
        GDF_MOD,
        GDF_POW,
        GDF_EQUAL,
        GDF_NOT_EQUAL,
        GDF_LESS,
        GDF_GREATER,
        GDF_LESS_EQUAL,
        GDF_GREATER_EQUAL,
        GDF_BITWISE_AND,
        GDF_BITWISE_OR,
        GDF_BITWISE_XOR,

    ctypedef enum gdf_unary_math_op:
        GDF_SIN,
        GDF_COS,
        GDF_TAN,
        GDF_ARCSIN,
        GDF_ARCCOS,
        GDF_ARCTAN,
        GDF_EXP,
        GDF_LOG,
        GDF_SQRT,
        GDF_CEIL,
        GDF_FLOOR,
        GDF_ABS,
        GDF_BIT_INVERT,

    ctypedef union gdf_data:
        char          si08
        short         si16
        int           si32
        long          si64
        float         fp32
        double        fp64
        gdf_date32    dt32
        gdf_date64    dt64
        gdf_timestamp tmst

    ctypedef struct gdf_scalar:
        gdf_data  data
        gdf_dtype dtype
        bool      is_valid

    cdef gdf_error gdf_count_nonzero_mask(gdf_valid_type * masks, int num_rows, int * count)

    cdef gdf_size_type gdf_column_sizeof()

    gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                              gdf_size_type size, gdf_dtype dtype)

    cdef gdf_error gdf_column_view_augmented(gdf_column *column,
                                             void *data,
                                             gdf_valid_type *valid,
                                             gdf_size_type size,
                                             gdf_dtype dtype,
                                             gdf_size_type null_count,
                                             gdf_dtype_extra_info extra_info)

    cdef gdf_error gdf_column_free(gdf_column *column)

    cdef gdf_error gdf_column_concat(gdf_column *output, gdf_column *columns_to_concat[], int num_columns)

    cdef gdf_error gdf_context_view(gdf_context *context,
                                    int flag_sorted,
                                    gdf_method flag_method,
                                    int flag_distinct,
                                    int flag_sort_result,
                                    int flag_sort_inplace)

    cdef const char * gdf_error_get_name(gdf_error errcode)

    cdef int gdf_cuda_last_error()
    cdef const char * gdf_cuda_error_string(int cuda_error)
    cdef const char * gdf_cuda_error_name(int cuda_error)

    cdef gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length)
    cdef void gdf_ipc_parser_open_recordbatches(gdf_ipc_parser_type *handle,
                                           const uint8_t *recordbatches,
                                           size_t length)

    cdef void gdf_ipc_parser_close(gdf_ipc_parser_type *handle)
    cdef int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle)
    cdef const char* gdf_ipc_parser_to_json(gdf_ipc_parser_type *handle)
    cdef const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle)
    cdef const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle)
    cdef int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle)

    cdef const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle)
    cdef const char *gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle)

    cdef gdf_radixsort_plan_type* gdf_radixsort_plan(size_t num_items, int descending,
                                            unsigned begin_bit, unsigned end_bit)
    cdef gdf_error gdf_radixsort_plan_setup(gdf_radixsort_plan_type *hdl,
                                       size_t sizeof_key, size_t sizeof_val)
    cdef gdf_error gdf_radixsort_plan_free(gdf_radixsort_plan_type *hdl)

    cdef gdf_error gdf_radixsort(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol)

    cdef gdf_segmented_radixsort_plan_type* gdf_segmented_radixsort_plan(size_t num_items, int descending,
        unsigned begin_bit, unsigned end_bit)
    cdef gdf_error gdf_segmented_radixsort_plan_setup(gdf_segmented_radixsort_plan_type *hdl,
    size_t sizeof_key, size_t sizeof_val)
    cdef gdf_error gdf_segmented_radixsort_plan_free(gdf_segmented_radixsort_plan_type *hdl)

    cdef gdf_error gdf_segmented_radixsort(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)

    cdef gdf_error gdf_inner_join(
                             gdf_column **left_cols,
                             int num_left_cols,
                             int left_join_cols[],
                             gdf_column **right_cols,
                             int num_right_cols,
                             int right_join_cols[],
                             int num_cols_to_join,
                             int result_num_cols,
                             gdf_column **result_cols,
                             gdf_column * left_indices,
                             gdf_column * right_indices,
                             gdf_context *join_context) except +

    cdef gdf_error gdf_left_join(
                             gdf_column **left_cols,
                             int num_left_cols,
                             int left_join_cols[],
                             gdf_column **right_cols,
                             int num_right_cols,
                             int right_join_cols[],
                             int num_cols_to_join,
                             int result_num_cols,
                             gdf_column **result_cols,
                             gdf_column * left_indices,
                             gdf_column * right_indices,
                             gdf_context *join_context) except +

    cdef gdf_error gdf_full_join(
                             gdf_column **left_cols,
                             int num_left_cols,
                             int left_join_cols[],
                             gdf_column **right_cols,
                             int num_right_cols,
                             int right_join_cols[],
                             int num_cols_to_join,
                             int result_num_cols,
                             gdf_column **result_cols,
                             gdf_column * left_indices,
                             gdf_column * right_indices,
                             gdf_context *join_context) except +

    cdef gdf_error gdf_hash_partition(int num_input_cols,
                                 gdf_column * input[],
                                 int columns_to_hash[],
                                 int num_cols_to_hash,
                                 int num_partitions,
                                 gdf_column * partitioned_output[],
                                 int partition_offsets[],
                                 gdf_hash_func hash)

    cdef gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output)

    cdef gdf_error gdf_unary_math(gdf_column *input, gdf_column *output, gdf_unary_math_op op)

    cdef gdf_error gdf_cast(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_binary_operation_s_v(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope)
    cdef gdf_error gdf_binary_operation_v_s(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope)
    cdef gdf_error gdf_binary_operation_v_v(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope)

    cdef gdf_error gdf_add_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_add_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_add_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_add_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_sub_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_sub_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_sub_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_sub_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_sub_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_mul_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_mul_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_mul_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_mul_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_mul_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_floordiv_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_floordiv_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_floordiv_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_floordiv_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_floordiv_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_div_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_div_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_div_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_gt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_gt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_gt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_gt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_gt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_gt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_ge_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ge_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ge_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ge_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ge_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ge_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_lt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_lt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_lt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_lt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_lt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_lt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_le_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_le_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_le_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_le_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_le_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_le_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_eq_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_eq_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_eq_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_eq_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_eq_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_eq_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_ne_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ne_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ne_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ne_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ne_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_ne_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_bitwise_and_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_bitwise_or_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_bitwise_xor_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_validity_and(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    
    cdef gdf_error gdf_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output)

    cdef gdf_size_type gdf_dtype_size(gdf_dtype dtype) except +

    cdef gdf_error gdf_hash_columns(gdf_column ** columns_to_hash, int num_columns, gdf_column * output_column, void * stream)

    cdef gdf_error get_column_byte_width(gdf_column * col, int * width)

    cdef gdf_error gdf_order_by(gdf_column** input_columns,
                                int8_t* asc_desc,
                                size_t num_inputs,
                                gdf_column* output_indices,
                                int flag_nulls_are_smallest)

    cdef gdf_error gdf_filter(size_t nrows,
                 gdf_column* cols,
                 size_t ncols,
                 void** d_cols,
                 int* d_types,
                 void** d_vals,
                 size_t* d_indx,
                 size_t* new_sz)

    cdef gdf_error gdf_group_by_sum(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt)

    cdef gdf_error gdf_group_by_min(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt)


    cdef gdf_error gdf_group_by_max(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt)


    cdef gdf_error gdf_group_by_avg(int ncols,
                               gdf_column** cols,
                               gdf_column* col_agg,
                               gdf_column* out_col_indices,
                               gdf_column** out_col_values,

                               gdf_column* out_col_agg,
                               gdf_context* ctxt)

    cdef gdf_error gdf_group_by_count(int ncols,
                                 gdf_column** cols,
                                 gdf_column* col_agg,
                                 gdf_column* out_col_indices,
                                 gdf_column** out_col_values,

                                 gdf_column* out_col_agg,
                                 gdf_context* ctxt)


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
                                   gdf_column* new_values)


    cdef gdf_error gdf_replace_nulls(gdf_column* col_out,
                                     const gdf_column* col_in)


    cdef gdf_error gdf_digitize(gdf_column* col,
                                gdf_column* bins,
                                bool right,
                                gdf_index_type* out_indices)

    cdef gdf_error gdf_from_dlpack(gdf_column** columns,
                                   gdf_size_type *num_columns,
                                   const DLManagedTensor * tensor) except +

    cdef gdf_error gdf_to_dlpack(DLManagedTensor *tensor,
                                 const gdf_column ** columns,
                                 gdf_size_type num_columns) except +

    cdef gdf_error gdf_nvtx_range_push(const char * const name, gdf_color color ) except +

    cdef gdf_error gdf_nvtx_range_push_hex(const char * const name, unsigned int color ) except +

    cdef gdf_error gdf_nvtx_range_pop() except +

