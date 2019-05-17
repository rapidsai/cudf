#include <cassert>
#include <thrust/fill.h>
#include <algorithm>
#include <tuple>

#include "cudf.h"
#include "types.hpp"
#include "copying.hpp"
#include "new_groupby.hpp"
#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/error_utils.hpp"
#include "aggregation_operations.hpp"
#include "groupby/hash_groupby.cuh"
#include "groupby/sort_groupby.cuh"
#include "string/nvcategory_util.hpp"
#include "table/device_table.cuh"


namespace{
  /* --------------------------------------------------------------------------*/
  /**
   * @brief Verifies that a set gdf_columns contain non-null data buffers, and are all
   * of the same size.
   *
   *
   * TODO: remove when null support added.
   *
   * Also ensures that the columns do not contain any null values
   *
   * @param[in] first Pointer to first gdf_column in set
   * @param[in] last Pointer to one past the last column in set
   *
   * @returns GDF_DATASET_EMPTY if a column contains a null data buffer,
   * GDF_COLUMN_SIZE_MISMATCH if the columns are not of equal length,
   */
  /* ----------------------------------------------------------------------------*/
  gdf_error verify_columns(gdf_column * cols[], int num_cols)
  {
    GDF_REQUIRE((nullptr != cols[0]), GDF_DATASET_EMPTY);

    gdf_size_type const required_size{cols[0]->size};

    for(int i = 0; i < num_cols; ++i)
    {
      GDF_REQUIRE(nullptr != cols[i], GDF_DATASET_EMPTY);
      GDF_REQUIRE(nullptr != cols[i]->data, GDF_DATASET_EMPTY);
      GDF_REQUIRE(required_size == cols[i]->size, GDF_COLUMN_SIZE_MISMATCH );

      // TODO Remove when null support for hash-based groupby is added
      GDF_REQUIRE(nullptr == cols[i]->valid || 0 == cols[i]->null_count, GDF_VALIDITY_UNSUPPORTED);
    }
    return GDF_SUCCESS;
  }
} // anonymous namespace

/* --------------------------------------------------------------------------*/
/**
 * @brief  Groupby operation for an arbitrary number of key columns and an
 * arbitrary number of aggregation columns.
 *
 * "Groupby" is a reduce-by-key operation where rows in one or more "key" columns
 * act as the keys and one or more "aggregation" columns hold the values that will
 * be reduced.
 *
 * The output of the operation is the set of key columns that hold all the unique keys
 * from the input key columns and a set of aggregation columns that hold the specified
 * reduction among all identical keys.
 *
 * @param[in] in_key_columns[] The input key columns
 * @param[in] num_key_columns The number of input columns to groupby
 * @param[in] in_aggregation_columns[] The columns that will be aggregated
 * @param[in] num_aggregation_columns The number of columns that will be aggregated
 * @param[in] agg_ops[] The aggregation operations to perform. The number of aggregation
 * operations must be equal to the number of aggregation columns, such that agg_op[i]
 * will be applied to in_aggregation_columns[i]
 * @param[in,out] out_key_columns[] Preallocated buffers for the output key columns
 * columns
 * @param[in,out] out_aggregation_columns[] Preallocated buffers for the output
 * aggregation columns
 * @param[in] options Structure that controls behavior of groupby operation, i.e.,
 * sort vs. hash-based implementation, whether or not the output will be sorted,
 * etc. See definition of gdf_context.
 *
 * @returns GDF_SUCCESS upon succesful completion. Otherwise appropriate error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_group_by(gdf_column* in_key_columns[],
                       int num_key_columns,
                       gdf_column* in_aggregation_columns[],
                       int num_aggregation_columns,
                       gdf_agg_op agg_ops[],
                       gdf_column* out_key_columns[],
                       gdf_column* out_aggregation_columns[],
                       gdf_context* options)
{

  // TODO: Remove when single pass multi-agg is implemented
  if(num_aggregation_columns > 1)
    assert(false && "Only 1 aggregation column currently supported.");

  // TODO: Remove when the `flag_method` member is removed from `gdf_context`
  if(GDF_SORT == options->flag_method)
    assert(false && "Sort-based groupby is no longer supported.");

  // Ensure inputs aren't null
  if( (0 == num_key_columns)
      || (0 == num_aggregation_columns)
      || (nullptr == in_key_columns)
      || (nullptr == in_aggregation_columns)
      || (nullptr == agg_ops)
      || (nullptr == out_key_columns)
      || (nullptr == out_aggregation_columns)
      || (nullptr == options))
  {
    return GDF_DATASET_EMPTY;
  }

  // Return immediately if inputs are empty
  GDF_REQUIRE(0 != in_key_columns[0]->size, GDF_SUCCESS);
  GDF_REQUIRE(0 != in_aggregation_columns[0]->size, GDF_SUCCESS);

  auto result = verify_columns(in_key_columns, num_key_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  result = verify_columns(in_aggregation_columns, num_aggregation_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_GROUPBY", GROUPBY_COLOR);


  bool sort_result = false;

  if( 0 != options->flag_sort_result){
    sort_result = true;
  }

  // Check that user is not trying to sum or avg string columns
  for(int aggregation_index = 0; aggregation_index < num_aggregation_columns; aggregation_index++){
    if(( agg_ops[aggregation_index] == GDF_SUM ||
       agg_ops[aggregation_index] == GDF_AVG ) &&
       in_aggregation_columns[aggregation_index]->dtype == GDF_STRING_CATEGORY){
      return GDF_UNSUPPORTED_DTYPE;
    }

  }

  // TODO: Only a single aggregator supported right now
  gdf_agg_op op{agg_ops[0]};

  switch(op)
  {
    case GDF_MAX:
      {
        gdf_error_code = gdf_group_by_hash<max_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_MIN:
      {
        gdf_error_code = gdf_group_by_hash<min_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_SUM:
      {
        gdf_error_code = gdf_group_by_hash<sum_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_COUNT:
      {
        gdf_error_code = gdf_group_by_hash<count_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_AVG:
      {
        gdf_error_code = gdf_group_by_hash_avg(num_key_columns,
                                               in_key_columns,
                                               in_aggregation_columns[0],
                                               out_key_columns,
                                               out_aggregation_columns[0]);

        break;
      }
    default:
      std::cerr << "Unsupported aggregation method for hash-based groupby." << std::endl;
      gdf_error_code = GDF_UNSUPPORTED_METHOD;
  }

  GDF_REQUIRE(GDF_SUCCESS == gdf_error_code, gdf_error_code);

  // The following code handles propogating an NVCategory into columns which are of type nvcategory
  for(int key_index = 0; key_index < num_key_columns; key_index++){
    if(out_key_columns[key_index]->dtype == GDF_STRING_CATEGORY){
      gdf_error_code = nvcategory_gather(out_key_columns[key_index],
                                         static_cast<NVCategory *>(in_key_columns[key_index]->dtype_info.category));
      GDF_REQUIRE(GDF_SUCCESS == gdf_error_code, gdf_error_code);
    }
  }
  for(int out_column_index = 0; out_column_index < num_aggregation_columns; out_column_index++){
    if(out_aggregation_columns[out_column_index]->dtype == GDF_STRING_CATEGORY){
      gdf_error_code = nvcategory_gather(out_aggregation_columns[out_column_index],
                                         static_cast<NVCategory *>(in_aggregation_columns[out_column_index]->dtype_info.category));
      GDF_REQUIRE(GDF_SUCCESS == gdf_error_code, gdf_error_code);
    }
  }

  POP_RANGE();

  return gdf_error_code;
}

rmm::device_vector<gdf_index_type>
gdf_unique_indices(cudf::table const& input_table, gdf_context const& context)
{
  gdf_size_type ncols = input_table.num_columns();
  gdf_size_type nrows = input_table.num_rows();

  rmm::device_vector<void*> d_cols(ncols);
  rmm::device_vector<int> d_types(ncols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  gdf_index_type* result_end;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<gdf_index_type> unique_indices(nrows);

  auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);
  auto device_input_table = device_table::create(input_table);
  bool nullable = device_input_table.get()->has_nulls();
  if (nullable){
    auto comp = row_equality_comparator<true>(*device_input_table, true);
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              comp);
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table, true);
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              comp);
  }

  unique_indices.resize(thrust::distance(unique_indices.data().get(), result_end));

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return unique_indices;
}

std::tuple<cudf::table, rmm::device_vector<gdf_index_type>> 
gdf_group_by_without_aggregations(cudf::table const& input_table,
                                  gdf_size_type num_key_cols,
                                  gdf_index_type const * key_col_indices,
                                  gdf_context* context)
{
  CUDF_EXPECTS(nullptr != key_col_indices, "key_col_indices is null");
  CUDF_EXPECTS(0 < num_key_cols, "number of key colums should be greater than zero");

  if (0 == input_table.num_rows()) {
    return std::make_tuple(cudf::table(), rmm::device_vector<gdf_index_type>());
  }

  gdf_size_type nrows = input_table.num_rows();
  
  // Allocate output columns
  cudf::table destination_table(nrows, cudf::column_dtypes(input_table), true);

  std::vector<gdf_column*> key_cols_vect(num_key_cols);
  std::transform(key_col_indices, key_col_indices+num_key_cols, key_cols_vect.begin(),
                  [&input_table] (gdf_index_type const index) { return const_cast<gdf_column*>(input_table.get_column(index)); });
  cudf::table key_col_table(key_cols_vect.data(), key_cols_vect.size());

  rmm::device_vector<gdf_size_type> sorted_indices(nrows);
  gdf_column sorted_indices_col;
  CUDF_TRY(gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()),
                          nullptr, nrows, GDF_INT32));

  if (context->flag_groupby_include_nulls || !cudf::has_nulls(key_col_table)){  // SQL style
    CUDF_TRY(gdf_order_by(key_col_table.begin(),
                          nullptr,
                          key_col_table.num_columns(),
                          &sorted_indices_col,
                          context));
  } else {  // Pandas style

    // Pandas style ignores groups that have nulls in their keys, so we want to filter them out.
    // We will create a bitmask (key_cols_bitmask) that represents if there is any null in any of they key columns.
    // We create a modified set of key columns (modified_key_col_table), where the first key column will take this bitmask (key_cols_bitmask)
    // Then if we set flag_null_sort_behavior = GDF_NULL_AS_LARGEST, then when we sort by the key columns, 
    // then all the rows where any of the key columns contained a null, these will be at the end of the sorted set.
    // Then we can figure out how many of those rows contained any nulls and adjust the size of our sorted data set 
    // to ignore the rows where there were any nulls in the key columns
    
    auto key_cols_bitmask = row_bitmask(key_col_table);

    gdf_column modified_fist_key_col; 
    modified_fist_key_col.data = key_cols_vect[0]->data;
    modified_fist_key_col.size = key_cols_vect[0]->size;
    modified_fist_key_col.dtype = key_cols_vect[0]->dtype;
    modified_fist_key_col.null_count = key_cols_vect[0]->null_count;
    modified_fist_key_col.valid = reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get());

    std::vector<gdf_column*> modified_key_cols_vect = key_cols_vect;
    modified_key_cols_vect[0] = &modified_fist_key_col;
    cudf::table modified_key_col_table(modified_key_cols_vect.data(), modified_key_cols_vect.size());

    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

    CUDF_TRY(gdf_order_by(modified_key_col_table.begin(),
                          nullptr,
                          modified_key_col_table.num_columns(),
                          &sorted_indices_col,
                          &temp_ctx));
    
    int valid_count;
    CUDF_TRY(gdf_count_nonzero_mask(reinterpret_cast<gdf_valid_type*>(key_cols_bitmask.data().get()),
                                    nrows, &valid_count));

    std::for_each(destination_table.begin(), destination_table.end(), 
                  [valid_count](gdf_column * col){ col->size = valid_count; });
  }

  // run gather operation to establish new order
  cudf::gather(&input_table, sorted_indices.data().get(), &destination_table);

  std::vector<gdf_column*> key_cols_vect_out(num_key_cols);
  std::transform(key_col_indices, key_col_indices+num_key_cols, key_cols_vect_out.begin(),
                  [&destination_table] (gdf_index_type const index) { return destination_table.get_column(index); });
  cudf::table key_col_sorted_table(key_cols_vect_out.data(), key_cols_vect_out.size());
  
  return std::make_tuple(destination_table, gdf_unique_indices(key_col_sorted_table, *context));
}


gdf_error gdf_group_by_sort(gdf_column* in_key_columns[],
                       int num_key_columns,
                       gdf_column* in_aggregation_columns[],
                       int num_aggregation_columns,
                       gdf_agg_op agg_ops[],
                       gdf_column* out_key_columns[],
                       gdf_column* out_aggregation_columns[],
                       gdf_context* options)
{

  // TODO: Remove when single pass multi-agg is implemented
  if(num_aggregation_columns > 1)
    assert(false && "Only 1 aggregation column currently supported.");

  // Ensure inputs aren't null
  if( (0 == num_key_columns)
      || (0 == num_aggregation_columns)
      || (nullptr == in_key_columns)
      || (nullptr == in_aggregation_columns)
      || (nullptr == agg_ops)
      || (nullptr == out_key_columns)
      || (nullptr == out_aggregation_columns)
      || (nullptr == options))
  {
    return GDF_DATASET_EMPTY;
  }

  // Return immediately if inputs are empty
  GDF_REQUIRE(0 != in_key_columns[0]->size, GDF_SUCCESS);
  GDF_REQUIRE(0 != in_aggregation_columns[0]->size, GDF_SUCCESS);

  auto result = verify_columns(in_key_columns, num_key_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  result = verify_columns(in_aggregation_columns, num_aggregation_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_GROUPBY_SORT", GROUPBY_COLOR);

  // INSTEAD OF: use options object
  // bool sort_result = false;
  // if( 0 != options->flag_sort_result){
  //   sort_result = true;
  // }
 
  // TODO: Only a single aggregator supported right now
  gdf_agg_op op{agg_ops[0]};

  switch(op)
  { 
    case GDF_MAX:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<max_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options);
        break;
      }
    case GDF_MIN:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<min_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options);
        break;
      }
    case GDF_SUM:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<sum_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options);
        break;
      }
    case GDF_COUNT:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<count_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options);
        break;
      }
    case GDF_COUNT_DISTINCT:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<count_distinct_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options);
        break;
      }
    case GDF_AVG:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort_avg(num_key_columns,
                                               in_key_columns,
                                               in_aggregation_columns[0],
                                               out_key_columns,
                                               out_aggregation_columns[0]);

        break;
      }
    default:
      std::cerr << "Unsupported aggregation method for sort-based groupby." << std::endl;
      gdf_error_code = GDF_UNSUPPORTED_METHOD;
  }

  POP_RANGE();

  return gdf_error_code;
}
