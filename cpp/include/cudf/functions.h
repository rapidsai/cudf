#pragma once

/** 
 * @brief  Start a NVTX range with predefined color.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 * 
 * @param[in] name The name of the NVTX range
 * @param[in] color The predefined gdf_color enum to use to color this range
 * 
 * @returns   
 */
gdf_error gdf_nvtx_range_push(char const * const name, gdf_color color );

/** 
 * @brief  Start a NVTX range with a custom ARGB color code.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * demarcates the begining of a user-defined range with a specified name and
 * color that will show up in the timeline view of nvvp/Nsight Systems. Can be
 * nested within other ranges.
 * 
 * @param[in] name The name of the NVTX range
 * @param[in] color The ARGB hex color code to use to color this range (e.g., 0xFF00FF00)
 * 
 * @returns   
 */
gdf_error gdf_nvtx_range_push_hex(char const * const name, unsigned int color );

/** 
 * @brief Ends the inner-most NVTX range.
 *
 * This function is useful only for profiling with nvvp or Nsight Systems. It
 * will demarcate the end of the inner-most range, i.e., the most recent call to
 * gdf_nvtx_range_push.
 * 
 * @returns   
 */
gdf_error gdf_nvtx_range_pop();


/**
 * Calculates the number of bytes to allocate for a column's validity bitmask
 *
 * For a column with a specified number of elements, returns the required size
 * in bytes of the validity bitmask to provide one bit per element.
 *
 * @note Note that this function assumes the bitmask needs to be allocated to be
 * padded to a multiple of 64 bytes
 * 
 * @note This function assumes that the size of gdf_valid_type is 1 byte
 *
 * @param[in] column_size The number of elements
 * @return the number of bytes necessary to allocate for validity bitmask
 */
gdf_size_type gdf_valid_allocation_size(gdf_size_type column_size);

/**
 * @brief Computes the number of `gdf_valid_type` elements required to provide
 * enough bits to represent the specified number of column elements.
 *
 * @note Note that this function assumes that the size of `gdf_valid_type` is 1
 * byte
 * @note This function is different gdf_valid_allocation_size
 * because gdf_valid_allocation_size returns the number of bytes required to
 * satisfy 64B padding. This function should be used when needing to access the
 * last `gdf_valid_type` element in the validity bitmask.
 *
 * @param[in] column_size the number of elements
 * @return The minimum number of `gdf_valid_type` elements to provide sufficient
 * bits to represent elements in a column of size @p column_size
 */
gdf_size_type gdf_num_bitmask_elements(gdf_size_type column_size);

/* column operations */

/** 
 * @brief Return the size of the gdf_column data type.
 *
 * @returns gdf_size_type Size of the gdf_column data type.
 */
gdf_size_type gdf_column_sizeof();

/** 
 * @brief Create a GDF column given data and validity bitmask pointers, size, and
 *        datatype
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 * 
 * @returns gdf_error returns GDF_SUCCESS upon successful creation.
 */
gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype);

/** 
 * @brief Create a GDF column given data and validity bitmask pointers, size, and
 *        datatype, and count of null (non-valid) elements
 *
 * @param[out] column The output column.
 * @param[in] data Pointer to data.
 * @param[in] valid Pointer to validity bitmask for the data.
 * @param[in] size Number of rows in the column.
 * @param[in] dtype Data type of the column.
 * @param[in] null_count The number of non-valid elements in the validity bitmask.
 * @param[in] extra_info see gdf_dtype_extra_info. Extra data for column description.
 * 
 * @returns gdf_error returns GDF_SUCCESS upon successful creation.
 */
gdf_error gdf_column_view_augmented(gdf_column *column, void *data, gdf_valid_type *valid,
                          gdf_size_type size, gdf_dtype dtype, gdf_size_type null_count,
                          gdf_dtype_extra_info extra_info);

/** 
 * @brief Free the CUDA device memory of a gdf_column
 *
 * @param[in,out] column Data and validity bitmask pointers of this column will be freed
 * 
 * @returns gdf_error GDF_SUCCESS or GDF_ERROR if there is an error freeing the data
 */
gdf_error gdf_column_free(gdf_column *column);

/**
 * @brief Concatenates multiple gdf_columns into a single, contiguous column,
 * including the validity bitmasks.
 * 
 * Note that input columns with nullptr validity masks are treated as if all
 * elements are valid.
 *
 * @param[out] output_column A column whose buffers are already allocated that
 *             will contain the concatenation of the input columns data and
 *             validity bitmasks
 * @param[in] columns_to_concat[] The columns to concatenate
 * @param[in] num_columns The number of columns to concatenate
 * 
 * @return gdf_error GDF_SUCCESS upon completion; GDF_DATASET_EMPTY if any data
 *         pointer is NULL, GDF_COLUMN_SIZE_MISMATCH if the output column size
 *         != the total size of the input columns; GDF_DTYPE_MISMATCH if the
 *         input columns have different datatypes.
 *
 */
gdf_error gdf_column_concat(gdf_column *output, gdf_column *columns_to_concat[], int num_columns);


/* context operations */

/**
 * @brief  Constructor for the gdf_context struct
 *
 * @param[out] context gdf_context being constructed
 * @param[in] flag_sorted Indicates if the input data is sorted. 0 = No, 1 = yes
 * @param[in] flag_method The method to be used for the operation (e.g., sort vs hash)
 * @param[in] flag_distinct For COUNT: DISTINCT = 1, else = 0
 * @param[in] flag_sort_result When method is GDF_HASH, 0 = result is not sorted, 1 = result is sorted
 * @param[in] flag_sort_inplace 0 = No sort in place allowed, 1 = else
 * @param[in] flag_null_sort_behavior GDF_NULL_AS_LARGEST = Nulls are treated as largest,
 *                                    GDF_NULL_AS_SMALLEST = Nulls are treated as smallest, 
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_context_view(gdf_context *context, int flag_sorted, gdf_method flag_method,
                           int flag_distinct, int flag_sort_result, int flag_sort_inplace, 
                           gdf_null_sort_behavior flag_null_sort_behavior);


/* error handling */

/**
 * @brief  Converts a gdf_error error code into a string
 *
 * @param[in] gdf_error
 *
 * @returns  name of the error
 */
const char * gdf_error_get_name(gdf_error errcode);

/**
 * @brief  returns the last error from a runtime call.
 *
 * @returns  last error from a runtime call.
 */
int gdf_cuda_last_error();

/**
 * @brief  returns the description string for an error code.
 *
 * @param[in] cuda error code
 *
 * @returns  description string for an error code.
 */
const char * gdf_cuda_error_string(int cuda_error);

/**
 * @brief  returns the string representation of an error code enum name.
 *
 * @param[in] cuda error code
 *
 * @returns  string representation of an error code enum name.
 */
const char * gdf_cuda_error_name(int cuda_error);


/* ipc */

/**
 * @brief  Opens a parser from a pyarrow RecordBatch schema
 *
 * @param[in] Pointer to a byte array containing the pyarrow RecordBatch schema
 * @param[in] Size of the byte array
 *
 * @returns Pointer to a parsing struct gdf_ipc_parser_type
 */
gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length);

/**
 * @brief  Opens a pyarrow RecordBatch bytearray
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 * @param[in] Pointer to a pyarrow RecordBatch bytearray
 * @param[in] Size of the byte array
 *
 * @returns
 */
void gdf_ipc_parser_open_recordbatches(gdf_ipc_parser_type *handle,
                                       const uint8_t *recordbatches,
                                       size_t length);

/**
 * @brief  Closes a parser from a pyarrow RecordBatch schema
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns void
 */
void gdf_ipc_parser_close(gdf_ipc_parser_type *handle);

/**
 * @brief  Checks for a failure in the parser
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns 1 if error
 */
int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle);

/**
 * @brief  returns parsed data as json
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns char* of parsed data as json
 */
const char* gdf_ipc_parser_to_json(gdf_ipc_parser_type *handle);

/**
 * @brief  Gets error from gdf_ipc_parser_type
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns Error message as char*
 */
const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle);

/**
 * @brief  Gets parsed data from gdf_ipc_parser_type
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns Pointer parsed data
 */
const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle);

/**
 * @brief  Gets data offset from gdf_ipc_parser_type
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns Data offset
 */
int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle);

/**
 * @brief  returns parsed schema as json
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns char* of parsed schema as json
 */
const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle);

/**
 * @brief  returns layout as json
 *
 * @param[in] Pointer to a parsing struct gdf_ipc_parser_type
 *
 * @returns char* of layout as json
 */
const char *gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle);


/* segmented sorting */

/**
 * @brief  Constructor for the gdf_segmented_radixsort_plan_type object
 *
 * @param[in] Number of items to sort
 * @param[in] Indicates if sort should be ascending or descending. 1 = Descending, 0 = Ascending
 * @param[in] The least-significant bit index (inclusive) needed for key comparison
 * @param[in] The most-significant bit index (exclusive) needed for key comparison (e.g., sizeof(unsigned int) * 8)
 *
 * @returns  gdf_segmented_radixsort_plan_type object pointer
 */
gdf_segmented_radixsort_plan_type* gdf_segmented_radixsort_plan(size_t num_items,
                                                                int descending,
                                                                unsigned begin_bit,
                                                                unsigned end_bit);

/**
 * @brief  Allocates device memory for the segmented radixsort
 *
 * @param[in] Segmented Radix sort plan
 * @param[in] sizeof data type of key
 * @param[in] sizeof data type of val
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_segmented_radixsort_plan_setup(gdf_segmented_radixsort_plan_type *hdl,
                                            size_t sizeof_key,
                                            size_t sizeof_val);

/**
 * @brief  Frees device memory used for the segmented radixsort
 *
 * @param[in] Segmented Radix sort plan
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_segmented_radixsort_plan_free(gdf_segmented_radixsort_plan_type *hdl);


/**
 * @brief  Performs a segmented radixsort on the key and value columns
 * 
 * The null_count of the keycol and valcol columns are expected to be 0
 * otherwise a GDF_VALIDITY_UNSUPPORTED error is returned.
 *
 * @param[in] Radix sort plan
 * @param[in] key gdf_column
 * @param[in] value gdf_column
 * @param[in] The number of segments that comprise the sorting data
 * @param[in] Pointer to the sequence of beginning offsets of length num_segments, such that d_begin_offsets[i] is the first element of the ith data segment in d_keys_* and d_values_*
 * @param[in] Pointer to the sequence of ending offsets of length num_segments, such that d_end_offsets[i]-1 is the last element of the ith data segment in d_keys_* and d_values_*. If d_end_offsets[i]-1 <= d_begin_offsets[i], the ith is considered empty.
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_segmented_radixsort(gdf_segmented_radixsort_plan_type *hdl,
                                  gdf_column *keycol, gdf_column *valcol,
                                  unsigned num_segments,
                                  unsigned *d_begin_offsets,
                                  unsigned *d_end_offsets);
// transpose
/**
 * @brief Transposes the table in_cols and copies to out_cols
 * 
 * @param[in] ncols Number of columns in in_cols
 * @param[in] in_cols[] Input table of (ncols) number of columns each of size (nrows)
 * @param[out] out_cols[] Preallocated output_table of (nrows) columns each of size (ncols)
 * @returns gdf_error GDF_SUCCESS if successful, else appropriate error code
 */
gdf_error gdf_transpose(gdf_size_type ncols,
                        gdf_column** in_cols,
                        gdf_column** out_cols);

// joins

/** 
 * @brief  Performs an inner join on the specified columns of two
 * dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 * 
 * @param[in] left_cols[] The columns of the left dataframe
 * @param[in] num_left_cols The number of columns in the left dataframe
 * @param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols[] The columns of the right dataframe
 * @param[in] num_right_cols The number of columns in the right dataframe
 * @param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @param[in] num_cols_to_join The total number of columns to join on
 * @param[in] result_num_cols The number of columns in the resulting dataframe
 * @param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
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

/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 * 
 * @param[in] left_cols[] The columns of the left dataframe
 * @param[in] num_left_cols The number of columns in the left dataframe
 * @param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols[] The columns of the right dataframe
 * @param[in] num_right_cols The number of columns in the right dataframe
 * @param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @param[in] num_cols_to_join The total number of columns to join on
 * @param[in] result_num_cols The number of columns in the resulting dataframe
 * @param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
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

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 * 
 * @param[in] left_cols[] The columns of the left dataframe
 * @param[in] num_left_cols The number of columns in the left dataframe
 * @param[in] left_join_cols[] The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols[] The columns of the right dataframe
 * @param[in] num_right_cols The number of columns in the right dataframe
 * @param[in] right_join_cols[] The column indices of columns from the right dataframe
 * to join on
 * @param[in] num_cols_to_join The total number of columns to join on
 * @param[in] result_num_cols The number of columns in the resulting dataframe
 * @param[out] gdf_column *result_cols[] If not nullptr, the dataframe that results from joining
 * the left and right tables on the specified columns
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns   GDF_SUCCESS if the join operation was successful, otherwise an appropriate
 * error code
 */
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

/** 
 * @brief Computes the hash values of the rows in the specified columns of the 
 * input columns and bins the hash values into the desired number of partitions. 
 * Rearranges the input columns such that rows with hash values in the same bin 
 * are contiguous.
 * 
 * @param[in] num_input_cols The number of columns in the input columns
 * @param[in] input[] The input set of columns
 * @param[in] columns_to_hash[] Indices of the columns in the input set to hash
 * @param[in] num_cols_to_hash The number of columns to hash
 * @param[in] num_partitions The number of partitions to rearrange the input rows into
 * @param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement 
 * of the input columns into the desired number of partitions
 * @param[out] partition_offsets Preallocated array the size of the number of
 * partitions. Where partition_offsets[i] indicates the starting position
 * of partition 'i'
 * @param[in] hash The hash function to use
 * 
 * @returns  If the operation was successful, returns GDF_SUCCESS
 */
gdf_error gdf_hash_partition(int num_input_cols, 
                             gdf_column * input[], 
                             int columns_to_hash[],
                             int num_cols_to_hash,
                             int num_partitions, 
                             gdf_column * partitioned_output[],
                             int partition_offsets[],
                             gdf_hash_func hash);


/* unary operators */

/* hashing */
/** --------------------------------------------------------------------------*
 * @brief Computes the hash value of each row in the input set of columns.
 *
 * @param[in] num_cols The number of columns in the input set
 * @param[in] input The list of columns whose rows will be hashed
 * @param[in] hash The hash function to use
 * @param[in] initial_hash_values Optional array in device memory specifying an initial hash value for each column
 * that will be combined with the hash of every element in the column. If this argument is `nullptr`,
 * then each element will be hashed as-is.
 * @param[out] output The hash value of each row of the input
 *
 * @returns    GDF_SUCCESS if the operation was successful, otherwise an
 *            appropriate error code.
 * ----------------------------------------------------------------------------**/
gdf_error gdf_hash(int num_cols,
                   gdf_column **input,
                   gdf_hash_func hash,
                   uint32_t *initial_hash_values,
                   gdf_column *output);


/**
 * @brief  Performs unary math op on all values in column
 *
 * The following math operations are supported:
 * sin - Computes trigonometric sine function
 * cos - Computes trigonometric cosine function
 * tan - Computes trigonometric tangent function
 * asin - Computes trigonometric arcsin function
 * acos - Computes trigonometric arccos function
 * atan - Computes trigonometric arctan function
 * exp - Computes e (Euler's number, 2.7182818...) raised to the given power arg
 * log - Computes the natural (base e) logarithm of arg
 * sqrt - Computes the square root
 * ceil - Computes the smallest integer value not less than arg
 * floor - Computes the largest integer value not greater than arg
 * 
 * All operations only supported on floating point types
 * 
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 * @param[in] gdf_unary_math_op operation to perform
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_unary_math(gdf_column *input, gdf_column *output, gdf_unary_math_op op);

/* casting */

/**
 * @brief  Casts data from dtype specified in input to dtype specified in output
 * 
 * The desired dtype for output should be set in output->dtype.
 * In case of conversion from GDF_DATE32/GDF_DATE64/GDF_TIMESTAMP to GDF_TIMESTAMP,
 * the time unit for output should be set in output->dtype_info.time_unit
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_cast(gdf_column *input, gdf_column *output);


/* datetime extract*/

/**
 * @brief  Extracts year from any date time type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output);

/**
 * @brief  Extracts month from any date time type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output);

/**
 * @brief  Extracts day from any date time type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output);

/**
 * @brief  Extracts hour from either GDF_DATE64 or GDF_TIMESTAMP type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output);

/**
 * @brief  Extracts minute from either GDF_DATE64 or GDF_TIMESTAMP type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output);

/**
 * @brief  Extracts second from either GDF_DATE64 or GDF_TIMESTAMP type and places results into a preallocated GDF_INT16 column
 *
 * @param[in] gdf_column of the input
 * @param[out] output gdf_column. The output memory needs to be preallocated
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output);


/*
 * gdf introspection utlities
 */

/* 
 Multi-Column SQL ops:
   WHERE (Filtering)
   ORDER-BY
   GROUP-BY
 */

/**
 * @brief  Performs SQL like GROUP BY with SUM aggregation
 *
 * @param[in] # columns
 * @param[in] input cols
 * @param[in] column to aggregate on
 * @param[out] if not null return indices of re-ordered rows
 * @param[out] if not null return the grouped-by columns (multi-gather based on indices, which are needed anyway)
 * @param[out] aggregation result
 * @param[in] struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_group_by_sum(int ncols,
                           gdf_column** cols,
                           gdf_column* col_agg,
                           gdf_column* out_col_indices,
                           gdf_column** out_col_values,
                           gdf_column* out_col_agg,
                           gdf_context* ctxt); 

/**
 * @brief  Performs SQL like GROUP BY with MIN aggregation
 *
 * @param[in] # columns
 * @param[in] input cols
 * @param[in] column to aggregate on
 * @param[out] if not null return indices of re-ordered rows
 * @param[out] if not null return the grouped-by columns (multi-gather based on indices, which are needed anyway)
 * @param[out] aggregation result
 * @param[in] struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_group_by_min(int ncols,
                           gdf_column** cols,
                           gdf_column* col_agg,
                           gdf_column* out_col_indices,
                           gdf_column** out_col_values, 
                           gdf_column* out_col_agg,
                           gdf_context* ctxt);

/**
 * @brief  Performs SQL like GROUP BY with MAX aggregation
 *
 * @param[in] # columns
 * @param[in] input cols
 * @param[in] column to aggregate on
 * @param[out] if not null return indices of re-ordered rows
 * @param[out] if not null return the grouped-by columns (multi-gather based on indices, which are needed anyway)
 * @param[out] aggregation result
 * @param[in] struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_group_by_max(int ncols,
                           gdf_column** cols,
                           gdf_column* col_agg,
                           gdf_column* out_col_indices,
                           gdf_column** out_col_values,
                           gdf_column* out_col_agg,
                           gdf_context* ctxt);

/**
 * @brief  Performs SQL like GROUP BY with AVG aggregation
 *
 * @param[in] # columns
 * @param[in] input cols
 * @param[in] column to aggregate on
 * @param[out] if not null return indices of re-ordered rows
 * @param[out] if not null return the grouped-by columns (multi-gather based on indices, which are needed anyway)
 * @param[out] aggregation result
 * @param[in] struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_group_by_avg(int ncols,
                           gdf_column** cols,
                           gdf_column* col_agg,
                           gdf_column* out_col_indices,
                           gdf_column** out_col_values,
                           gdf_column* out_col_agg,
                           gdf_context* ctxt);

/**
 * @brief  Performs SQL like GROUP BY with COUNT aggregation
 *
 * @param[in] # columns
 * @param[in] input cols
 * @param[in] column to aggregate on
 * @param[out] if not null return indices of re-ordered rows
 * @param[out] if not null return the grouped-by columns (multi-gather based on indices, which are needed anyway)
 * @param[out] aggregation result
 * @param[in] struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_group_by_count(int ncols,
                             gdf_column** cols,
                             gdf_column* col_agg,
                             gdf_column* out_col_indices,
                             gdf_column** out_col_values,      
                             gdf_column* out_col_agg,
                             gdf_context* ctxt);

/**
 * @brief  Computes exact quantile
 * computes quantile as double. This function works with arithmetic colum.
 *
 * @param[in] input column
 * @param[in] precision: type of quantile method calculation
 * @param[in] requested quantile in [0,1]
 * @param[out] result the result as double. The type can be changed in future
 * @param[in] struct with additional info
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_quantile_exact(gdf_column* col_in,
                            gdf_quantile_method prec,
                            double q,
                            gdf_scalar*  result,
                            gdf_context* ctxt);

/**
 * @brief  Computes approximate quantile
 * computes quantile with the same type as @p col_in.
 * This function works with arithmetic colum.
 *
 * @param[in] input column
 * @param[in] requested quantile in [0,1]
 * @param[out] result quantile, with the same type as @p col_in
 * @param[in] struct with additional info
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error gdf_quantile_approx(gdf_column* col_in,
                              double q,
                              gdf_scalar*  result,
                              gdf_context* ctxt);


/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @param[in]  input_columns Array of gdf_columns
 * @param[in]  asc_desc Device array of sort order types for each column
 *                     (0 is ascending order and 1 is descending). If NULL
 *                     is provided defaults to ascending order for evey column.
 * @param[in]  num_inputs # columns
 * @param[out] output_indices Pre-allocated gdf_column to be filled with sorted indices
 * @param[in]  context  The options for controlling treatment of nulls
 *             context->flag_null_sort_behavior
 *                        GDF_NULL_AS_LARGEST = Nulls are treated as largest, 
 *                        GDF_NULL_AS_SMALLEST = Nulls are treated as smallest, 
 * 
 * @returns GDF_SUCCESS upon successful completion
 */
gdf_error gdf_order_by(gdf_column const* const* input_columns,
                       int8_t*      asc_desc,
                       size_t       num_inputs,
                       gdf_column*  output_indices,
                       gdf_context * context);

/**
 * @brief Finds the indices of the bins in which each value of the column
 * belongs.
 *
 * For `x` in `col`, if `right == false` this function finds
 * `i` such that `bins[i-1] <= x < bins[i]`. If `right == true`, it will find `i`
 * such that `bins[i - 1] < x <= bins[i]`. Finally, if `x < bins[0]` or
 * `x > bins[num_bins - 1]`, it sets the index to `0` or `num_bins`, respectively.
 *
 * NOTE: This function does not handle null values and will return an error if `col`
 * or `bins` contain any.
 *
 * @param[in] col gdf_column with the values to be binned
 * @param[in] bins gdf_column of ascending bin boundaries
 * @param[in] right Whether the intervals should include the left or right bin edge
 * @param[out] out_indices Output device array of same size as `col`
 * to be filled with bin indices
 *
 * @returns GDF_SUCCESS upon successful completion, otherwise an
 *         appropriate error code.
 */
gdf_error gdf_digitize(gdf_column* col,
                       gdf_column* bins,   // same type as col
                       bool right,
                       gdf_index_type out_indices[]);

// forward declaration for DLPack functions below
// This approach is necessary to satisfy CFFI
struct DLManagedTensor;
typedef struct DLManagedTensor DLManagedTensor_;

/**
 * @brief Convert a DLPack DLTensor into gdf_column(s)
 *
 * 1D and 2D tensors are supported. This function makes copies
 * of the input DLPack data into the created output columns.
 * 
 * Note: currently only supports column-major ("Fortran" order) memory layout
 * Therefore row-major tensors should be transposed before calling 
 * this function
 * TODO: provide a parameter to select row- or column-major ordering (row major
 * input will require a transpose)
 *
 * Note: this function does NOT call the input tensor's deleter currently
 * because the caller of this function may still need it. Also, it is often
 * packaged in a PyCapsule on the Python side, which (in the case of Cupy)
 * may be set up to call the deleter in its own destructor
 *
 * @param[out] columns The output column(s)
 * @param[out] num_columns The number of gdf_columns in columns
 * @param[in] tensor The input DLPack DLTensor
 * @return gdf_error GDF_SUCCESS if conversion is successful
 */
gdf_error gdf_from_dlpack(gdf_column** columns,
                          gdf_size_type *num_columns,
                          DLManagedTensor_ const * tensor);

/**
 * @brief Convert an array of gdf_column(s) into a DLPack DLTensor
 *
 * 1D and 2D tensors are supported. This function allocates the DLPack tensor 
 * data and copies the data from the input column(s) into the tensor.
 * 
 * Note: currently only supports column-major ("Fortran" order) memory layout
 * Therefore the output of this function should be transposed if row-major is
 * needed.
 * TODO: provide a parameter to select row- or column-major ordering (row major
 * output will require a transpose)
 *
 * @param[out] tensor The output DLTensor
 * @param[in] columns An array of pointers to gdf_column 
 * @param[in] num_columns The number of input columns
 * @return gdf_error GDF_SUCCESS if conversion is successful
 */
gdf_error gdf_to_dlpack(DLManagedTensor_ *tensor,
                        gdf_column const * const * columns,
                        gdf_size_type num_columns);
