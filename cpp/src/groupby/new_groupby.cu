#include <cassert>
#include <thrust/fill.h>
#include <algorithm>

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

  //Check that user is not trying to sum or avg string columns
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
  
  //The following code handles propogating an NVCategory into columns which are of type nvcategory
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



gdf_error gdf_unique_indices(gdf_size_type num_data_cols,
                             gdf_column** data_cols_in,
									           gdf_index_type* unique_indices,
                             gdf_size_type* num_unique_indices, 
									           gdf_context* context)
{

  gdf_size_type nrows = data_cols_in[0]->size;
  // setup for reduce by key  
  bool const have_nulls{ std::any_of(data_cols_in, data_cols_in + num_data_cols, [](gdf_column * col){ return col->null_count > 0;}) };

  rmm::device_vector<void*> d_cols(num_data_cols); 
  rmm::device_vector<int> d_types(num_data_cols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  bool nulls_are_smallest = context->flag_null_sort_behavior == GDF_NULL_AS_SMALLEST;

  gdf_index_type* result_end;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);
  if (have_nulls){

    rmm::device_vector<gdf_valid_type*> d_valids(num_data_cols);
    gdf_valid_type** d_valids_data = d_valids.data().get();

    soa_col_info(data_cols_in, num_data_cols, d_col_data, d_valids_data, d_col_types);
    
    LesserRTTI<gdf_size_type> comp(d_col_data, d_valids_data, d_col_types, nullptr, num_data_cols, nulls_are_smallest);
    
    auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              unique_indices,
                              [comp]  __device__(gdf_size_type key1, gdf_size_type key2){
                              return comp.equal_with_nulls(key1, key2);
                            });

  } else {

    soa_col_info(data_cols_in, num_data_cols, d_col_data, nullptr, d_col_types);
    
    LesserRTTI<gdf_size_type> comp(d_col_data, nullptr, d_col_types, nullptr, num_data_cols, nulls_are_smallest);

    auto counting_iter = thrust::make_counting_iterator<gdf_size_type>(0);
    
    result_end = thrust::unique_copy(exec, counting_iter, counting_iter+nrows, 
                              unique_indices,
                              [comp]  __device__(gdf_size_type key1, gdf_size_type key2){
                              return comp.equal(key1, key2);  
                            });
  }

  gdf_size_type new_sz = thrust::distance(unique_indices, result_end);
  *num_unique_indices = new_sz;
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream); 

  return GDF_SUCCESS;

}



gdf_error gdf_group_by_without_aggregations(gdf_size_type num_data_cols,
                                            gdf_column** data_cols_in,
                                            gdf_size_type num_key_cols,
                                            gdf_index_type const * key_col_indices,
                                            gdf_column** data_cols_out,
                                            gdf_index_type* group_start_indices,
                                            gdf_size_type* num_group_start_indices, 
                                            gdf_context* context)      
{
  GDF_REQUIRE((nullptr != data_cols_in), GDF_DATASET_EMPTY);
  GDF_REQUIRE((nullptr != data_cols_in[0]), GDF_DATASET_EMPTY);
  GDF_REQUIRE((nullptr != data_cols_out), GDF_DATASET_EMPTY);
  GDF_REQUIRE((nullptr != data_cols_out[0]), GDF_DATASET_EMPTY);
  GDF_REQUIRE((num_data_cols > 0), GDF_DATASET_EMPTY);
  GDF_REQUIRE((num_key_cols > 0), GDF_DATASET_EMPTY);
  GDF_REQUIRE((nullptr != key_col_indices), GDF_DATASET_EMPTY);
  

  gdf_size_type nrows = data_cols_in[0]->size;

  // setup for order by call
  bool group_by_keys_contain_nulls = false;
  std::vector<gdf_column*> orderby_cols_vect(num_key_cols);
  for (gdf_size_type i = 0; i < num_key_cols; i++){
    orderby_cols_vect[i] = data_cols_in[key_col_indices[i]];
    group_by_keys_contain_nulls = group_by_keys_contain_nulls || orderby_cols_vect[i]->null_count > 0;
  }

  rmm::device_vector<gdf_size_type> sorted_indices(nrows);
  gdf_column sorted_indices_col;
  gdf_error status = gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()), 
                            nullptr, nrows, GDF_INT32);
  GDF_REQUIRE(GDF_SUCCESS == status, status);

  if (context->flag_groupby_include_nulls || !group_by_keys_contain_nulls){  // SQL style
  // run order by and get new sort indexes
    status = gdf_order_by(&orderby_cols_vect[0],             //input columns
                          nullptr,
                          num_key_cols,                //number of columns in the first parameter (e.g. number of columsn to sort by)
                          &sorted_indices_col,            //a gdf_column that is pre allocated for storing sorted indices
                          context);
    GDF_REQUIRE(GDF_SUCCESS == status, status);

    // run gather operation to establish new order
    // run gather operation to establish new order
    cudf::table table_in(data_cols_in, num_data_cols);
    cudf::table table_out(data_cols_out, num_data_cols);
    
    cudf::gather(&table_in, sorted_indices.data().get(), &table_out);
    
    for (gdf_size_type i = 0; i < num_key_cols; i++){
      orderby_cols_vect[i] = data_cols_out[key_col_indices[i]];
    }

    status = gdf_unique_indices(num_key_cols, &orderby_cols_vect[0], group_start_indices, num_group_start_indices, context);

    return status;
  } else {  // Pandas style

    auto flag_null_sort_behavior = context->flag_null_sort_behavior;
    context->flag_null_sort_behavior = GDF_NULL_AS_LARGEST_FOR_MULTISORT; // overide behaviour to filter out the nulls

    // run order by and get new sort indexes
    status = gdf_order_by(&orderby_cols_vect[0],             //input columns
                          nullptr,
                          num_key_cols,                //number of columns in the first parameter (e.g. number of columsn to sort by)
                          &sorted_indices_col,            //a gdf_column that is pre allocated for storing sorted indices
                          context);
    GDF_REQUIRE(GDF_SUCCESS == status, status);

    // lets filter out all the nulls in the group by key column by:
    // we will take the data which has been sorted such that the nulls in the group by keys are all last
    // then using the gdf_table's property of row-validity mask we can count how many rows have 
    // a null in the group by keys and use that to resize the data
    std::unique_ptr< gdf_table<gdf_size_type> > group_by_keys_table{new gdf_table<gdf_size_type>{num_key_cols, &orderby_cols_vect[0]}};
    int valid_count;
    status = group_by_keys_table->get_num_valid_rows(valid_count);
    GDF_REQUIRE(GDF_SUCCESS == status, status);

    for (gdf_size_type i = 0; i < num_data_cols; i++)    {
      data_cols_in[i]->size = valid_count;
      data_cols_out[i]->size = valid_count;
    }
    
    // run gather operation to establish new order
    cudf::table table_in(data_cols_in, num_data_cols);
    cudf::table table_out(data_cols_out, num_data_cols);
    
    cudf::gather(&table_in, sorted_indices.data().get(), &table_out);
    
    for (gdf_size_type i = 0; i < num_key_cols; i++){
      orderby_cols_vect[i] = data_cols_out[key_col_indices[i]];
    }
   
    context->flag_null_sort_behavior = flag_null_sort_behavior;

    status = gdf_unique_indices(num_key_cols, &orderby_cols_vect[0], group_start_indices, num_group_start_indices, context);

    return status;
  }
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
