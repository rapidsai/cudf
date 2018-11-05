# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from numpy cimport uint8_t, int64_t, int32_t, int16_t, int8_t

# Utility functions to build gdf_columns, gdf_context and error handling

cpdef _get_ctype_ptr(obj)
cpdef _get_column_data_ptr(obj)
cpdef _get_column_valid_ptr(obj)

cdef gdf_column* column_view_from_column(col)
cdef gdf_column* column_view_from_NDArrays(size, data, mask,
                                           dtype, null_count)

cdef gdf_context* create_context_view(flag_sorted, method, flag_distinct,
                                      flag_sort_result, flag_sort_inplace)

cpdef check_gdf_error(errcode)

# Import cudf.h header to import all functions
# First version of bindings has no changes to the cudf.h header, so this file
# mirrors the structure in cpp/include

cdef extern from "gdf.h" nogil:

    ctypedef size_t gdf_size_type
    ctypedef gdf_size_type gdf_index_type
    ctypedef unsigned char gdf_valid_type
    ctypedef long    gdf_date64
    ctypedef int     gdf_date32
    ctypedef int     gdf_category

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

    # ctypedef struct gdf_column_{
    #     void *data
    #     gdf_valid_type *valid
    #     gdf_size_type size
    #     gdf_dtype dtype
    #     gdf_size_type null_count
    #     gdf_dtype_extra_info dtype_info
    #     char *          col_name
    #  gdf_column
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


    ctypedef enum gdf_comparison_operator:
        GDF_EQUALS,
        GDF_NOT_EQUALS,
        GDF_LESS_THAN,
        GDF_LESS_THAN_OR_EQUALS,
        GDF_GREATER_THAN,
        GDF_GREATER_THAN_OR_EQUALS


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


        #pragma once

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Start a NVTX range with predefined color.
    #  *
    #  * This function is useful only for profiling with nvvp or Nsight Systems. It
    #  * demarcates the begining of a user-defined range with a specified name and
    #  * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
    #  * nested within other ranges.
    #  *
    #  * @Param name The name of the NVTX range
    #  * @Param color The predefined gdf_color enum to use to color this range
    #  *
    #  * @Returns
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_nvtx_range_push(char  *  name, gdf_color color )




    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Start a NVTX range with a custom ARGB color code.
    #  *
    #  * This function is useful only for profiling with nvvp or Nsight Systems. It
    #  * demarcates the begining of a user-defined range with a specified name and
    #  * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
    #  * nested within other ranges.
    #  *
    #  * @Param name The name of the NVTX range
    #  * @Param color The ARGB hex color code to use to color this range (e.g., 0xFF00FF00)
    #  *
    #  * @Returns
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_nvtx_range_push_hex(char * name, unsigned int color )


    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis Ends the inner-most NVTX range.
    #  *
    #  * This function is useful only for profiling with nvvp or Nsight Systems. It
    #  * will demarcate the end of the inner-most range, i.e., the most recent call to
    #  * gdf_nvtx_range_push.
    #  *
    #  * @Returns
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_nvtx_range_pop()

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Counts the number of valid bits in the mask that corresponds to
    #  * the specified number of rows.
    #  *
    #  * @Param[in] masks Array of gdf_valid_types with enough bits to represent
    #  * num_rows number of rows
    #  * @Param[in] num_rows The number of rows represented in the bit-validity mask.
    #  * @Param[out] count The number of valid rows in the mask
    #  *
    #  * @Returns  GDF_SUCCESS upon successful completion.
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_count_nonzero_mask(gdf_valid_type * masks, int num_rows, int * count)

    # /* column operations */

    cdef gdf_size_type gdf_column_sizeof()

    gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                              gdf_size_type size, gdf_dtype dtype)

    cdef gdf_error gdf_column_view_augmented(gdf_column *column, void *data, gdf_valid_type *valid,
                              gdf_size_type size, gdf_dtype dtype, gdf_size_type null_count)

    cdef gdf_error gdf_column_free(gdf_column *column)

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Concatenates the gdf_columns into a single, contiguous column,
    #  * including the validity bitmasks
    #  *
    #  * @Param[out] output A column whose buffers are already allocated that will
    #  * @Param[in] columns_to_conat[] The columns to concatenate
    #  * @Param[in] num_columns The number of columns to concatenate
    #   * contain the concatenation of the input columns
    #  *
    #  * @Returns GDF_SUCCESS upon successful completion
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_column_concat(gdf_column *output, gdf_column *columns_to_concat[], int num_columns)

    # /* context operations */

    cdef gdf_error gdf_context_view(gdf_context *context,
                                    int flag_sorted,
                                    gdf_method flag_method,
                                    int flag_distinct,
                                    int flag_sort_result,
                                    int flag_sort_inplace)

    # /* error handling */

    cdef const char * gdf_error_get_name(gdf_error errcode)

    cdef int gdf_cuda_last_error()
    cdef const char * gdf_cuda_error_string(int cuda_error)
    cdef const char * gdf_cuda_error_name(int cuda_error)

    # /* ipc */

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


    # /* sorting */
    cdef gdf_radixsort_plan_type* gdf_radixsort_plan(size_t num_items, int descending,
                                            unsigned begin_bit, unsigned end_bit)
    cdef gdf_error gdf_radixsort_plan_setup(gdf_radixsort_plan_type *hdl,
                                       size_t sizeof_key, size_t sizeof_val)
    cdef gdf_error gdf_radixsort_plan_free(gdf_radixsort_plan_type *hdl)

    # /*
    #  * The following function performs a sort on the key and value columns.
    #  */
    cdef gdf_error gdf_radixsort_i8(gdf_radixsort_plan_type *hdl,
                               gdf_column *keycol,
                               gdf_column *valcol)
    cdef gdf_error gdf_radixsort_i32(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol)
    cdef gdf_error gdf_radixsort_i64(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol)
    cdef gdf_error gdf_radixsort_f32(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol)
    cdef gdf_error gdf_radixsort_f64(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol)
    cdef gdf_error gdf_radixsort_generic(gdf_radixsort_plan_type *hdl,
                                    gdf_column *keycol,
                                    gdf_column *valcol)

    # /* segmented sorting */
    cdef gdf_segmented_radixsort_plan_type* gdf_segmented_radixsort_plan(size_t num_items, int descending,
        unsigned begin_bit, unsigned end_bit)
    cdef gdf_error gdf_segmented_radixsort_plan_setup(gdf_segmented_radixsort_plan_type *hdl,
    size_t sizeof_key, size_t sizeof_val)
    cdef gdf_error gdf_segmented_radixsort_plan_free(gdf_segmented_radixsort_plan_type *hdl)

    # /*
    # * The following function performs a sort on the key and value columns.
    # */
    cdef gdf_error gdf_segmented_radixsort_i8(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)
    cdef gdf_error gdf_segmented_radixsort_i32(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)
    cdef gdf_error gdf_segmented_radixsort_i64(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)
    cdef gdf_error gdf_segmented_radixsort_f32(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)
    cdef gdf_error gdf_segmented_radixsort_f64(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)
    cdef gdf_error gdf_segmented_radixsort_generic(gdf_segmented_radixsort_plan_type *hdl,
                                         gdf_column *keycol, gdf_column *valcol,
                                         unsigned num_segments,
                                         unsigned *d_begin_offsets,
                                         unsigned *d_end_offsets)

    # // joins


    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Performs an inner join on the specified columns of two
    #  * dataframes (left, right)
    #  *
    #  * @Param[in] left_cols[] The columns of the left dataframe
    #  * @Param[in] num_left_cols The number of columns in the left dataframe
    #  * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
    #  * to join on
    #  * @Param[in] right_cols[] The columns of the right dataframe
    #  * @Param[in] num_right_cols The number of columns in the right dataframe
    #  * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
    #  * to join on
    #  * @Param[in] num_cols_to_join The total number of columns to join on
    #  * @Param[in] result_num_cols The number of columns in the resulting dataframe
    #  * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
    #  * the left and right tables on the specified columns
    #  * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
    #  * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
    #  * @Param[in] join_context The context to use to control how the join is performed,e.g.,
    #  * sort vs hash based implementation
    #  *
    #  * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
    #  * error code
    #  */
    # /* ----------------------------------------------------------------------------*/
    gdf_error gdf_inner_join(
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
                             gdf_context *join_context)

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Performs a left join (also known as left outer join) on the
    #  * specified columns of two dataframes (left, right)
    #  *
    #  * @Param[in] left_cols[] The columns of the left dataframe
    #  * @Param[in] num_left_cols The number of columns in the left dataframe
    #  * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
    #  * to join on
    #  * @Param[in] right_cols[] The columns of the right dataframe
    #  * @Param[in] num_right_cols The number of columns in the right dataframe
    #  * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
    #  * to join on
    #  * @Param[in] num_cols_to_join The total number of columns to join on
    #  * @Param[in] result_num_cols The number of columns in the resulting dataframe
    #  * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
    #  * the left and right tables on the specified columns
    #  * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
    #  * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
    #  * @Param[in] join_context The context to use to control how the join is performed,e.g.,
    #  * sort vs hash based implementation
    #  *
    #  * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
    #  * error code
    #  */
    # /* ----------------------------------------------------------------------------*/
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
                             gdf_context *join_context)

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Performs a full join (also known as full outer join) on the
    #  * specified columns of two dataframes (left, right)
    #  *
    #  * @Param[in] left_cols[] The columns of the left dataframe
    #  * @Param[in] num_left_cols The number of columns in the left dataframe
    #  * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
    #  * to join on
    #  * @Param[in] right_cols[] The columns of the right dataframe
    #  * @Param[in] num_right_cols The number of columns in the right dataframe
    #  * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
    #  * to join on
    #  * @Param[in] num_cols_to_join The total number of columns to join on
    #  * @Param[in] result_num_cols The number of columns in the resulting dataframe
    #  * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
    #  * the left and right tables on the specified columns
    #  * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
    #  * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
    #  * @Param[in] join_context The context to use to control how the join is performed,e.g.,
    #  * sort vs hash based implementation
    #  *
    #  * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
    #  * error code
    #  */
    # /* ----------------------------------------------------------------------------*/
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
                             gdf_context *join_context)

    # /* partioning */

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @brief Computes the hash values of the rows in the specified columns of the
    #  * input columns and bins the hash values into the desired number of partitions.
    #  * Rearranges the input columns such that rows with hash values in the same bin
    #  * are contiguous.
    #  *
    #  * @Param[in] num_input_cols The number of columns in the input columns
    #  * @Param[in] input[] The input set of columns
    #  * @Param[in] columns_to_hash[] Indices of the columns in the input set to hash
    #  * @Param[in] num_cols_to_hash The number of columns to hash
    #  * @Param[in] num_partitions The number of partitions to rearrange the input rows into
    #  * @Param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement
    #  * of the input columns into the desired number of partitions
    #  * @Param[out] partition_offsets Preallocated array the size of the number of
    #  * partitions. Where partition_offsets[i] indicates the starting position
    #  * of partition 'i'
    #  * @Param[in] hash The hash function to use
    #  *
    #  * @Returns  If the operation was successful, returns GDF_SUCCESS
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_hash_partition(int num_input_cols,
                                 gdf_column * input[],
                                 int columns_to_hash[],
                                 int num_cols_to_hash,
                                 int num_partitions,
                                 gdf_column * partitioned_output[],
                                 int partition_offsets[],
                                 gdf_hash_func hash)

    # /* prefixsum */

    cdef gdf_error gdf_prefixsum_generic(gdf_column *inp, gdf_column *out, int inclusive)
    cdef gdf_error gdf_prefixsum_i8(gdf_column *inp, gdf_column *out, int inclusive)
    cdef gdf_error gdf_prefixsum_i32(gdf_column *inp, gdf_column *out, int inclusive)
    cdef gdf_error gdf_prefixsum_i64(gdf_column *inp, gdf_column *out, int inclusive)


    # /* unary operators */

    # /* hashing */

    # /* --------------------------------------------------------------------------*/
    # /**
    #  * @Synopsis  Computes the hash value of each row in the input set of columns.
    #  *
    #  * @Param num_cols The number of columns in the input set
    #  * @Param input The list of columns whose rows will be hashed
    #  * @Param hash The hash function to use
    #  * @Param output The hash value of each row of the input
    #  *
    #  * @Returns   GDF_SUCCESS if the operation was successful, otherwise an appropriate
    #  * error code
    #  */
    # /* ----------------------------------------------------------------------------*/
    cdef gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output)

    # /* trig */

    cdef gdf_error gdf_sin_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_sin_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_sin_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cos_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cos_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cos_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_tan_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_tan_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_tan_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_asin_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_asin_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_asin_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_acos_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_acos_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_acos_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_atan_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_atan_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_atan_f64(gdf_column *input, gdf_column *output)

    # /* exponential */

    cdef gdf_error gdf_exp_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_exp_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_exp_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_log_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_log_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_log_f64(gdf_column *input, gdf_column *output)

    # /* power */

    cdef gdf_error gdf_sqrt_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_sqrt_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_sqrt_f64(gdf_column *input, gdf_column *output)


    # /* rounding */

    cdef gdf_error gdf_ceil_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_ceil_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_ceil_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_floor_generic(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_floor_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_floor_f64(gdf_column *input, gdf_column *output)

    # /* casting */

    cdef gdf_error gdf_cast_generic_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_f32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_f32(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_f64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_f64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_i8(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_i8(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_i32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_i32(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_i64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_i64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_date32(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_date32(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i8_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i32_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_i64_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f32_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_f64_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date32_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_date64_to_date64(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_cast_timestamp_to_date64(gdf_column *input, gdf_column *output)

    cdef gdf_error gdf_cast_generic_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_i8_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_i32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_i64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_f32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_f64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_date32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_date64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
    cdef gdf_error gdf_cast_timestamp_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)

    # /* datetime extract*/
    cdef gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output)
    cdef gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output)


    # /* binary operators */

    # /* arith */

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

    # /* logical */

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

    # /* bitwise */

    cdef gdf_error gdf_bitwise_and_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_and_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    cdef gdf_error gdf_bitwise_or_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_or_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)


    # /*
    #  * Filtering and comparison operators
    #  */

    cdef gdf_error gdf_bitwise_xor_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
    cdef gdf_error gdf_bitwise_xor_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    # /* validity */

    cdef gdf_error gdf_validity_and(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    # /* reductions

    # The following reduction functions use the result array as a temporary working
    # space.  Use gdf_reduce_optimal_output_size() to get its optimal size.
    # */

    cdef unsigned int gdf_reduce_optimal_output_size()

    cdef gdf_error gdf_sum_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size)

    cdef gdf_error gdf_product_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_product_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_product_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_product_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_product_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_product_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size)

    # /* sum squared is useful for variance implementation */
    cdef gdf_error gdf_sum_squared_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_squared_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_sum_squared_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size)


    cdef gdf_error gdf_min_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_min_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_min_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_min_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_min_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_min_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size)

    cdef gdf_error gdf_max_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_max_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_max_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_max_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_max_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size)
    cdef gdf_error gdf_max_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size)




    # /*
    #  * Filtering and comparison operators
    #  */


    # //These compare every value on the left hand side to a static value and return a stencil in output which will have 1 when the comparison operation returns 1 and 0 otherwise
    cdef gdf_error gpu_comparison_static_i8(gdf_column *lhs, int8_t value, gdf_column *output,gdf_comparison_operator operation)
    cdef gdf_error gpu_comparison_static_i16(gdf_column *lhs, int16_t value, gdf_column *output,gdf_comparison_operator operation)
    cdef gdf_error gpu_comparison_static_i32(gdf_column *lhs, int32_t value, gdf_column *output,gdf_comparison_operator operation)
    cdef gdf_error gpu_comparison_static_i64(gdf_column *lhs, int64_t value, gdf_column *output,gdf_comparison_operator operation)
    cdef gdf_error gpu_comparison_static_f32(gdf_column *lhs, float value, gdf_column *output,gdf_comparison_operator operation)
    cdef gdf_error gpu_comparison_static_f64(gdf_column *lhs, double value, gdf_column *output,gdf_comparison_operator operation)

    # //allows you two compare two columns against each other using a comparison operation, retunrs a stencil like functions above
    cdef gdf_error gpu_comparison(gdf_column *lhs, gdf_column *rhs, gdf_column *output,gdf_comparison_operator operation)

    # //takes a stencil and uses it to compact a colum e.g. remove all values for which the stencil = 0
    cdef gdf_error gpu_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output)

    cdef gdf_error gpu_concat(gdf_column *lhs, gdf_column *rhs, gdf_column *output)

    # /*
    #  * Hashing
    #  */
    # //class cudaStream_t

    cdef gdf_error gpu_hash_columns(gdf_column ** columns_to_hash, int num_columns, gdf_column * output_column, void * stream)

    # /*
    #  * gdf introspection utlities
    #  */

    cdef gdf_error get_column_byte_width(gdf_column * col, int * width)

    # /*
    #  Multi-Column SQL ops:
    #    WHERE (Filtering)
    #    ORDER-BY
    #    GROUP-BY
    #  */
    cdef gdf_error gdf_order_by(size_t nrows,
                   gdf_column* cols,
                   size_t ncols,
                   void** d_cols,
                   int* d_types,
                   size_t* d_indx)

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

    cdef gdf_error gdf_quantile_exact(   gdf_column*         col_in,
                                    gdf_quantile_method prec,
                                    double              q,
                                    void*               t_erased_res,


                                    gdf_context*        ctxt)

    cdef gdf_error gdf_quantile_aprrox(  gdf_column*  col_in,
                                    double       q,
                                    void*        t_erased_res,
                                    gdf_context* ctxt)


