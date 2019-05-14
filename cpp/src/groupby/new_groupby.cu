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
#include "string/nvcategory_util.hpp"
#include "sqls/sqls_rtti_comp.h"

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

  bool nulls_are_smallest = (context.flag_null_sort_behavior == GDF_NULL_AS_SMALLEST);

  gdf_index_type* result_end;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<gdf_index_type> unique_indices(nrows);

  if (cudf::has_nulls(input_table)){
    rmm::device_vector<gdf_valid_type*> d_valids(ncols);
    gdf_valid_type** d_valids_data = d_valids.data().get();

    soa_col_info(input_table.begin(), ncols, d_col_data, d_valids_data, d_col_types);

    LesserRTTI<gdf_size_type> comp(d_col_data, d_valids_data, d_col_types, nullptr, ncols, nulls_are_smallest);

    auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);

    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              [comp]  __device__(gdf_size_type key1, gdf_size_type key2){
                              return comp.equal_with_nulls(key1, key2);
                            });

  } else {
    soa_col_info(input_table.begin(), ncols, d_col_data, nullptr, d_col_types);

    LesserRTTI<gdf_size_type> comp(d_col_data, nullptr, d_col_types, nullptr, ncols, nulls_are_smallest);

    auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);

    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows,
                              unique_indices.data().get(),
                              [comp]  __device__(gdf_size_type key1, gdf_size_type key2){
                              return comp.equal(key1, key2);
                            });
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
    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST_FOR_MULTISORT;

    CUDF_TRY(gdf_order_by(key_col_table.begin(),
                          nullptr,
                          key_col_table.num_columns(),
                          &sorted_indices_col,
                          &temp_ctx));

    // lets filter out all the nulls in the group by key column by:
    // we will take the data which has been sorted such that the nulls in the group by keys are all last
    // then using row_bitmask we can count how many rows have a null in the group by keys and use that 
    // to resize the data
    auto orderby_cols_bitmask = row_bitmask(key_col_table);
    int valid_count;
    CUDF_TRY(gdf_count_nonzero_mask(reinterpret_cast<gdf_valid_type*>(orderby_cols_bitmask.data().get()),
                                    nrows, &valid_count));

    std::for_each(destination_table.begin(), destination_table.end(), 
                  [valid_count](gdf_column * col){ col->size = valid_count; });
  }

  // run gather operation to establish new order
  cudf::gather(&input_table, sorted_indices.data().get(), &destination_table);

  std::transform(key_col_indices, key_col_indices+num_key_cols, key_cols_vect.begin(),
                  [&destination_table] (gdf_index_type const index) { return destination_table.get_column(index); });
  cudf::table key_col_sorted_table(key_cols_vect.data(), key_cols_vect.size());
  
  return std::make_tuple(destination_table, gdf_unique_indices(key_col_sorted_table, *context));
}
