#pragma once

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Start a NVTX range with predefined color.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 * 
 * @Param name The name of the NVTX range
 * @Param color The predefined gdf_color enum to use to color this range
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_push(char const * const name, gdf_color color );




/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Start a NVTX range with a custom ARGB color code.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 * 
 * @Param name The name of the NVTX range
 * @Param color The ARGB hex color code to use to color this range (e.g., 0xFF00FF00)
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_push_hex(char const * const name, unsigned int color );


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Ends the inner-most NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * will demarcate the end of the inner-most range, i.e., the most recent call to
 * gdf_nvtx_range_push.
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_nvtx_range_pop();

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Counts the number of valid bits in the mask that corresponds to
 * the specified number of rows.
 * 
 * @Param[in] masks Array of gdf_valid_types with enough bits to represent
 * num_rows number of rows
 * @Param[in] num_rows The number of rows represented in the bit-validity mask.
 * @Param[out] count The number of valid rows in the mask
 * 
 * @Returns  GDF_SUCCESS upon successful completion. 
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_count_nonzero_mask(gdf_valid_type const * masks, int num_rows, int * count);

/* column operations */

gdf_size_type gdf_column_sizeof();

gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype);

gdf_error gdf_column_view_augmented(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype, gdf_size_type null_count);

gdf_error gdf_column_free(gdf_column *column);

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Concatenates the gdf_columns into a single, contiguous column,
 * including the validity bitmasks
 * 
 * @Param[out] output A column whose buffers are already allocated that will 
 * @Param[in] columns_to_conat[] The columns to concatenate
 * @Param[in] num_columns The number of columns to concatenate
  * contain the concatenation of the input columns
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_column_concat(gdf_column *output, gdf_column *columns_to_concat[], int num_columns);

/* context operations */

gdf_error gdf_context_view(gdf_context *context, int flag_sorted, gdf_method flag_method,
                           int flag_distinct, int flag_sort_result, int flag_sort_inplace);

/* error handling */

const char * gdf_error_get_name(gdf_error errcode);

int gdf_cuda_last_error();
const char * gdf_cuda_error_string(int cuda_error);
const char * gdf_cuda_error_name(int cuda_error);

/* ipc */

gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length);
void gdf_ipc_parser_open_recordbatches(gdf_ipc_parser_type *handle,
                                       const uint8_t *recordbatches,
                                       size_t length);

void gdf_ipc_parser_close(gdf_ipc_parser_type *handle);
int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle);
const char* gdf_ipc_parser_to_json(gdf_ipc_parser_type *handle);
const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle);
const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle);
int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle);

const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle) ;
const char *gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle) ;


/* sorting */
gdf_radixsort_plan_type* gdf_radixsort_plan(size_t num_items, int descending,
                                        unsigned begin_bit, unsigned end_bit);
gdf_error gdf_radixsort_plan_setup(gdf_radixsort_plan_type *hdl,
                                   size_t sizeof_key, size_t sizeof_val);
gdf_error gdf_radixsort_plan_free(gdf_radixsort_plan_type *hdl);

/*
 * The following function performs a sort on the key and value columns.
 */
gdf_error gdf_radixsort_i8(gdf_radixsort_plan_type *hdl,
                           gdf_column *keycol,
                           gdf_column *valcol);
gdf_error gdf_radixsort_i32(gdf_radixsort_plan_type *hdl,
                            gdf_column *keycol,
                            gdf_column *valcol);
gdf_error gdf_radixsort_i64(gdf_radixsort_plan_type *hdl,
                            gdf_column *keycol,
                            gdf_column *valcol);
gdf_error gdf_radixsort_f32(gdf_radixsort_plan_type *hdl,
                            gdf_column *keycol,
                            gdf_column *valcol);
gdf_error gdf_radixsort_f64(gdf_radixsort_plan_type *hdl,
                            gdf_column *keycol,
                            gdf_column *valcol);
gdf_error gdf_radixsort_generic(gdf_radixsort_plan_type *hdl,
                                gdf_column *keycol,
                                gdf_column *valcol);

/* segmented sorting */
gdf_segmented_radixsort_plan_type* gdf_segmented_radixsort_plan(size_t num_items, int descending,
    unsigned begin_bit, unsigned end_bit);
gdf_error gdf_segmented_radixsort_plan_setup(gdf_segmented_radixsort_plan_type *hdl,
size_t sizeof_key, size_t sizeof_val);
gdf_error gdf_segmented_radixsort_plan_free(gdf_segmented_radixsort_plan_type *hdl);

/*
* The following function performs a sort on the key and value columns.
*/
gdf_error gdf_segmented_radixsort_i8(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);
gdf_error gdf_segmented_radixsort_i32(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);
gdf_error gdf_segmented_radixsort_i64(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);
gdf_error gdf_segmented_radixsort_f32(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);
gdf_error gdf_segmented_radixsort_f64(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);
gdf_error gdf_segmented_radixsort_generic(gdf_segmented_radixsort_plan_type *hdl,
                                     gdf_column *keycol, gdf_column *valcol,
                                     unsigned num_segments,
                                     unsigned *d_begin_offsets,
                                     unsigned *d_end_offsets);

// joins


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Performs an inner join on the specified columns of two
 * dataframes (left, right)
 * 
 * @Param[in] left_cols[] The columns of the left dataframe
 * @Param[in] num_left_cols The number of columns in the left dataframe
 * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @Param[in] right_cols[] The columns of the right dataframe
 * @Param[in] num_right_cols The number of columns in the right dataframe
 * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @Param[in] num_cols_to_join The total number of columns to join on
 * @Param[in] result_num_cols The number of columns in the resulting dataframe
 * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @Param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
/* ----------------------------------------------------------------------------*/
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
                         gdf_context *join_context);

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 * 
 * @Param[in] left_cols[] The columns of the left dataframe
 * @Param[in] num_left_cols The number of columns in the left dataframe
 * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @Param[in] right_cols[] The columns of the right dataframe
 * @Param[in] num_right_cols The number of columns in the right dataframe
 * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @Param[in] num_cols_to_join The total number of columns to join on
 * @Param[in] result_num_cols The number of columns in the resulting dataframe
 * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @Param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_left_join(
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
                         gdf_context *join_context);

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 * 
 * @Param[in] left_cols[] The columns of the left dataframe
 * @Param[in] num_left_cols The number of columns in the left dataframe
 * @Param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @Param[in] right_cols[] The columns of the right dataframe
 * @Param[in] num_right_cols The number of columns in the right dataframe
 * @Param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @Param[in] num_cols_to_join The total number of columns to join on
 * @Param[in] result_num_cols The number of columns in the resulting dataframe
 * @Param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @Param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @Param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @Param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @Returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_full_join(
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
                         gdf_context *join_context);

/* partioning */

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes the hash values of the rows in the specified columns of the 
 * input columns and bins the hash values into the desired number of partitions. 
 * Rearranges the input columns such that rows with hash values in the same bin 
 * are contiguous.
 * 
 * @Param[in] num_input_cols The number of columns in the input columns
 * @Param[in] input[] The input set of columns
 * @Param[in] columns_to_hash[] Indices of the columns in the input set to hash
 * @Param[in] num_cols_to_hash The number of columns to hash
 * @Param[in] num_partitions The number of partitions to rearrange the input rows into
 * @Param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement 
 * of the input columns into the desired number of partitions
 * @Param[out] partition_offsets Preallocated array the size of the number of
 * partitions. Where partition_offsets[i] indicates the starting position
 * of partition 'i'
 * @Param[in] hash The hash function to use
 * 
 * @Returns  If the operation was successful, returns GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_hash_partition(int num_input_cols, 
                             gdf_column * input[], 
                             int columns_to_hash[],
                             int num_cols_to_hash,
                             int num_partitions, 
                             gdf_column * partitioned_output[],
                             int partition_offsets[],
                             gdf_hash_func hash);

/* prefixsum */

gdf_error gdf_prefixsum_generic(gdf_column *inp, gdf_column *out, int inclusive);
gdf_error gdf_prefixsum_i8(gdf_column *inp, gdf_column *out, int inclusive);
gdf_error gdf_prefixsum_i32(gdf_column *inp, gdf_column *out, int inclusive);
gdf_error gdf_prefixsum_i64(gdf_column *inp, gdf_column *out, int inclusive);


/* unary operators */

/* hashing */

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the hash value of each row in the input set of columns.
 * 
 * @Param num_cols The number of columns in the input set
 * @Param input The list of columns whose rows will be hashed
 * @Param hash The hash function to use
 * @Param output The hash value of each row of the input
 * 
 * @Returns   GDF_SUCCESS if the operation was successful, otherwise an appropriate
 * error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output);

/* trig */

gdf_error gdf_sin_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_sin_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_sin_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_cos_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_cos_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cos_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_tan_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_tan_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_tan_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_asin_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_asin_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_asin_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_acos_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_acos_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_acos_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_atan_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_atan_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_atan_f64(gdf_column *input, gdf_column *output);

/* exponential */

gdf_error gdf_exp_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_exp_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_exp_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_log_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_log_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_log_f64(gdf_column *input, gdf_column *output);

/* power */

gdf_error gdf_sqrt_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_sqrt_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_sqrt_f64(gdf_column *input, gdf_column *output);


/* rounding */

gdf_error gdf_ceil_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_ceil_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_ceil_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_floor_generic(gdf_column *input, gdf_column *output);
gdf_error gdf_floor_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_floor_f64(gdf_column *input, gdf_column *output);

/* casting */

gdf_error gdf_cast_generic_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_f32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_f32(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_f64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_f64(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_i8(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_i8(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_i32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_i32(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_i64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_i64(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_date32(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_date32(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i8_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i32_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_i64_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f32_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_f64_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date32_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_date64_to_date64(gdf_column *input, gdf_column *output);
gdf_error gdf_cast_timestamp_to_date64(gdf_column *input, gdf_column *output);

gdf_error gdf_cast_generic_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_i8_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_i32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_i64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_f32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_f64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_date32_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_date64_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);
gdf_error gdf_cast_timestamp_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit);

/* datetime extract*/
gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output);
gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output);
gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output);
gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output);
gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output);
gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output);


/* binary operators */

/* arith */

gdf_error gdf_add_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_add_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_sub_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_sub_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_mul_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_mul_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_floordiv_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_floordiv_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_div_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_div_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_div_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/* logical */

gdf_error gdf_gt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_gt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_ge_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ge_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_lt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_lt_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_le_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_le_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_eq_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_eq_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_ne_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_ne_f64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/* bitwise */

gdf_error gdf_bitwise_and_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_and_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_and_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_and_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

gdf_error gdf_bitwise_or_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_or_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_or_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_or_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);


/*
 * Filtering and comparison operators
 */

gdf_error gdf_bitwise_xor_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_xor_i8(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_xor_i32(gdf_column *lhs, gdf_column *rhs, gdf_column *output);
gdf_error gdf_bitwise_xor_i64(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/* validity */

gdf_error gdf_validity_and(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/* reductions

The following reduction functions use the result array as a temporary working
space.  Use gdf_reduce_optimal_output_size() to get its optimal size.
*/

unsigned int gdf_reduce_optimal_output_size();

gdf_error gdf_sum_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size);

gdf_error gdf_product_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_product_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_product_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_product_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_product_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_product_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size);

/* sum squared is useful for variance implementation */
gdf_error gdf_sum_squared_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_squared_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_sum_squared_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size);


gdf_error gdf_min_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_min_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_min_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_min_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_min_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_min_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size);

gdf_error gdf_max_generic(gdf_column *col, void *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_max_f64(gdf_column *col, double *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_max_f32(gdf_column *col, float *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_max_i64(gdf_column *col, int64_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_max_i32(gdf_column *col, int32_t *dev_result, gdf_size_type dev_result_size);
gdf_error gdf_max_i8(gdf_column *col, int8_t *dev_result, gdf_size_type dev_result_size);




/*
 * Filtering and comparison operators
 */


//These compare every value on the left hand side to a static value and return a stencil in output which will have 1 when the comparison operation returns 1 and 0 otherwise
gdf_error gpu_comparison_static_i8(gdf_column *lhs, int8_t value, gdf_column *output,gdf_comparison_operator operation);
gdf_error gpu_comparison_static_i16(gdf_column *lhs, int16_t value, gdf_column *output,gdf_comparison_operator operation);
gdf_error gpu_comparison_static_i32(gdf_column *lhs, int32_t value, gdf_column *output,gdf_comparison_operator operation);
gdf_error gpu_comparison_static_i64(gdf_column *lhs, int64_t value, gdf_column *output,gdf_comparison_operator operation);
gdf_error gpu_comparison_static_f32(gdf_column *lhs, float value, gdf_column *output,gdf_comparison_operator operation);
gdf_error gpu_comparison_static_f64(gdf_column *lhs, double value, gdf_column *output,gdf_comparison_operator operation);

//allows you two compare two columns against each other using a comparison operation, retunrs a stencil like functions above
gdf_error gpu_comparison(gdf_column *lhs, gdf_column *rhs, gdf_column *output,gdf_comparison_operator operation);

//takes a stencil and uses it to compact a colum e.g. remove all values for which the stencil = 0
gdf_error gpu_apply_stencil(gdf_column *lhs, gdf_column * stencil, gdf_column * output);

gdf_error gpu_concat(gdf_column *lhs, gdf_column *rhs, gdf_column *output);

/*
 * Hashing
 */
//class cudaStream_t;

gdf_error gpu_hash_columns(gdf_column ** columns_to_hash, int num_columns, gdf_column * output_column, void * stream);

/*
 * gdf introspection utlities
 */

gdf_error get_column_byte_width(gdf_column * col, int * width);

/* 
 Multi-Column SQL ops:
   WHERE (Filtering)
   ORDER-BY
   GROUP-BY
 */
gdf_error gdf_order_by(size_t nrows,     //in: # rows
		       gdf_column* cols, //in: host-side array of gdf_columns
		       size_t ncols,     //in: # cols
		       void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
		       int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
		       size_t* d_indx);  //out: device-side array of re-rdered row indices

gdf_error gdf_filter(size_t nrows,     //in: # rows
		     gdf_column* cols, //in: host-side array of gdf_columns
		     size_t ncols,     //in: # cols
		     void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
		     int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
		     void** d_vals,    //in: device-side array of values to filter against (type-erased)
		     size_t* d_indx,   //out: device-side array of row indices that remain after filtering
		     size_t* new_sz);  //out: host-side # rows that remain after filtering

gdf_error gdf_group_by_sum(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt);           //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct

gdf_error gdf_group_by_min(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct


gdf_error gdf_group_by_max(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct


gdf_error gdf_group_by_avg(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct

gdf_error gdf_group_by_count(int ncols,                    // # columns
                             gdf_column** cols,            //input cols
                             gdf_column* col_agg,          //column to aggregate on
                             gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                             gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                             gdf_column* out_col_agg,      //aggregation result
                             gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct

gdf_error gdf_quantile_exact(	gdf_column*         col_in,       //input column;
                                gdf_quantile_method prec,         //precision: type of quantile method calculation
                                double              q,            //requested quantile in [0,1]
                                void*               t_erased_res, //result; for <exact> should probably be double*; it's void* because
                                                                  //(1) for uniformity of interface with <approx>;
                                                                  //(2) for possible types bigger than double, in the future;
                                gdf_context*        ctxt);        //context info

gdf_error gdf_quantile_aprrox(	gdf_column*  col_in,       //input column;
                                double       q,            //requested quantile in [0,1]
                                void*        t_erased_res, //type-erased result of same type as column;
                                gdf_context* ctxt);        //context info
