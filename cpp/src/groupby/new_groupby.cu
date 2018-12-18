#include <cassert>
#include "cudf.h"
#include "new_groupby.hpp"
#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/error_utils.h"
#include "aggregation_operations.hpp"
#include "groupby/hash_groupby.cuh"

#include <thrust/fill.h>

#include <../tests/utilities/cudf_test_utils.cuh>
#include <iostream>

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
   * @Param[in] first Pointer to first gdf_column in set
   * @Param[in] last Pointer to one past the last column in set
   * 
   * @Returns GDF_DATASET_EMPTY if a column contains a null data buffer, 
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
 * @Param[in] in_key_columns[] The input key columns
 * @Param[in] num_key_columns The number of input columns to groupby
 * @Param[in] in_aggregation_columns[] The columns that will be aggregated
 * @Param[in] num_aggregation_columns The number of columns that will be aggregated
 * @Param[in] agg_ops[] The aggregation operations to perform. The number of aggregation
 * operations must be equal to the number of aggregation columns, such that agg_op[i]
 * will be applied to in_aggregation_columns[i]
 * @Param[in,out] out_key_columns[] Preallocated buffers for the output key columns
 * columns
 * @Param[in,out] out_aggregation_columns[] Preallocated buffers for the output 
 * aggregation columns
 * @Param[in] options Structure that controls behavior of groupby operation, i.e.,
 * sort vs. hash-based implementation, whether or not the output will be sorted,
 * etc. See definition of gdf_context.
 * 
 * @Returns GDF_SUCCESS upon succesful completion. Otherwise appropriate error code
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

  POP_RANGE();

  return gdf_error_code;
}


// thrust::device_vector set to use rmmAlloc and rmmFree.
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

/* --------------------------------------------------------------------------*/
  /**
   * @brief Given a set of columns, of which a subset of these are defined to be the group by columns,
   * all columns are sorted by the group by columns and returned, along with a column containing
   * a list of the start indices of each group
   *
   * @Param[in] The number of columns in the dataset
   * @Param[in] The input columns in the dataset
   * @Param[in] The number of columns to be grouping by
   * @Param[in] The host column indices of the input dataset that will be grouped by
   * @Param[out] The dataset sorted by the group by columns (needs to be pre-allocated)
   * @Param[out] A column containing the starting indices of each group. Indices based off of new sort order. (needs to be pre-allocated)
   * @Param[in] Flag indicating if nulls are smaller (0) or larger (1) than non nulls for the sort operation
   *
   * @Returns gdf_error with error code on failure, otherwise GDF_SUCESS
   */
  /* ----------------------------------------------------------------------------*/
gdf_error gdf_group_by_wo_aggregations(int num_data_cols,
                           	   	   	   gdf_column** data_cols_in,
									   int num_groupby_cols,
									   int * groupby_col_indices,
									   gdf_column** data_cols_out,
									   gdf_column* group_start_indices,
									   int nulls_are_smallest = 0)
{
// TODO ASSERTS: num_groupby_cols > 0

  int32_t nrows = data_cols_in[0]->size;

  // setup for order by call
  std::vector<gdf_column*> orderby_cols_vect(num_groupby_cols);
  for (int i = 0; i < num_groupby_cols; i++){
    orderby_cols_vect[i] = data_cols_in[groupby_col_indices[i]];
  }

  rmm::device_vector<int32_t> sorted_indices(nrows);
  gdf_column sorted_indices_col;
  gdf_error status = gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()), 
                            nullptr, nrows, GDF_INT32);
  if (status != GDF_SUCCESS)
    return status;

// run order by and get new sort indexes
  status = gdf_order_by(&orderby_cols_vect[0],             //input columns
                        nullptr,
                        num_groupby_cols,                //number of columns in the first parameter (e.g. number of columsn to sort by)
                        &sorted_indices_col,            //a gdf_column that is pre allocated for storing sorted indices
                        0);  //flag to indicate if nulls are to be considered smaller than non-nulls or viceversa
  if (status != GDF_SUCCESS)
    return status;

  // run gather operation to establish new order
  std::unique_ptr< gdf_table<int32_t> > table_in{new gdf_table<int32_t>{num_data_cols, data_cols_in}};
  std::unique_ptr< gdf_table<int32_t> > table_out{new gdf_table<int32_t>{num_data_cols, data_cols_out}};

  status = table_in->gather<int32_t>(sorted_indices, *table_out.get());
  if (status != GDF_SUCCESS)
    return status;

  // status = gdf_group_start_indices(num_data_cols, data_cols_out, num_groupby_cols,
	// 								   groupby_col_indices, group_start_indices, nulls_are_smallest);

// setup for reduce by key  
  bool have_nulls = false;
  for (int i = 0; i < num_groupby_cols; i++) {
    if (data_cols_in[i]->null_count > 0) {
      have_nulls = true;
      break;
    }
  }

  Vector<void*> d_cols(num_groupby_cols); 
  Vector<int> d_types(num_groupby_cols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  int32_t* result_end;
  auto exec = rmm::exec_policy();
  if (have_nulls){

    Vector<gdf_valid_type*> d_valids(num_groupby_cols);
    gdf_valid_type** d_valids_data = d_valids.data().get();

    soa_col_info(data_cols_out, num_groupby_cols, d_col_data, d_valids_data, d_col_types);
    
    LesserRTTI<int32_t> comp(d_col_data, d_valids_data, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);
    
    auto counting_iter = thrust::make_counting_iterator<int32_t>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              (int32_t*)group_start_indices->data,
                              [comp] __host__ __device__(int32_t key1, int32_t key2){
                              return comp.equal_with_nulls(key1, key2);
                            });

  } else {

    soa_col_info(*data_cols_out, num_groupby_cols, d_col_data, d_col_types);
    
    LesserRTTI<int32_t> comp(d_col_data, nullptr, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);

    auto counting_iter = thrust::make_counting_iterator<int32_t>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              (int32_t*)group_start_indices->data,
                              [comp] __host__ __device__(int32_t key1, int32_t key2){
                              return comp.equal(key1, key2);  
                            });
  }

  size_t new_sz = thrust::distance((int32_t*)group_start_indices->data, result_end);
  group_start_indices->size = new_sz;

  return status;
}


/* --------------------------------------------------------------------------*/
  /**
   * @brief Given a set of columns, of which a subset of these are defined to be the group by columns,
   * all input data is assumed to already be sorted by these group by columns. 
   * This function calculates a list of the start indices of each group
   *
   * @Param[in] The number of columns in the dataset (assumed to already be sorted)
   * @Param[in] The input columns in the dataset
   * @Param[in] The number of columns to be grouping by
   * @Param[in] The column indices of the input dataset that will be grouped by
   * @Param[out] A column containing the starting indices of each group. Indices based off of new sort order. (needs to be pre-allocated)
   *
   * @Returns gdf_error with error code on failure, otherwise GDF_SUCESS
   */
  /* ----------------------------------------------------------------------------*/
gdf_error gdf_group_start_indices(int num_data_cols,
                           	   	   	   gdf_column** data_cols_in,
									   int num_groupby_cols,
									   int * groupby_col_indices,
									   gdf_column* group_start_indices,
									   int nulls_are_smallest = 0)
{

  int32_t nrows = data_cols_in[0]->size;
  // setup for reduce by key  
  bool have_nulls = false;
  for (int i = 0; i < num_groupby_cols; i++) {
    if (data_cols_in[i]->null_count > 0) {
      have_nulls = true;
      break;
    }
  }

  Vector<void*> d_cols(num_groupby_cols); 
  Vector<int> d_types(num_groupby_cols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  int32_t* result_end;
  auto exec = rmm::exec_policy();
  if (have_nulls){

    Vector<gdf_valid_type*> d_valids(num_groupby_cols);
    gdf_valid_type** d_valids_data = d_valids.data().get();

    soa_col_info(data_cols_in, num_groupby_cols, d_col_data, d_valids_data, d_col_types);
    
    LesserRTTI<int32_t> comp(d_col_data, d_valids_data, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);
    
    auto counting_iter = thrust::make_counting_iterator<int32_t>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              (int32_t*)group_start_indices->data,
                              [comp] __host__ __device__(int32_t key1, int32_t key2){
                              return comp.equal_with_nulls(key1, key2);
                            });

  } else {

    soa_col_info(*data_cols_in, num_groupby_cols, d_col_data, d_col_types);
    
    LesserRTTI<int32_t> comp(d_col_data, nullptr, d_col_types, nullptr, num_groupby_cols, nulls_are_smallest);

    auto counting_iter = thrust::make_counting_iterator<int32_t>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              (int32_t*)group_start_indices->data,
                              [comp] __host__ __device__(int32_t key1, int32_t key2){
                              return comp.equal(key1, key2);  
                            });
  }

  size_t new_sz = thrust::distance((int32_t*)group_start_indices->data, result_end);
  group_start_indices->size = new_sz;

  return GDF_SUCCESS;

}
