#ifndef SORT_GROUPBY_H
#define SORT_GROUPBY_H
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include "cudf.h"
#include "utilities/error_utils.hpp"

#include "groupby_sort_compute_api.h"
#include "aggregation_operations.hpp"

#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"

namespace group_by_sort{ 


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Calls the Sort Based group by compute API to compute the groupby with 
 * aggregation.
 * 
 * @Param ncols, in_groupby_columns The input groupby table
 * @Param in_aggregation_column The input aggregation column
 * @Param out_groupby_columns The output groupby table
 * @Param out_aggregation_column The output aggregation column
 * @Param ctxt Flag to optionally sort the output
 * @tparam aggregation_type  The type of the aggregation column
 * @tparam op A binary functor that implements the aggregation operation
 * 
 * @Returns On failure, returns appropriate error code. Otherwise, GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
template <typename aggregation_type, 
          template <typename T> class op,
          typename size_type>
gdf_error typed_groupby(size_type num_groupby_cols,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_aggregation_column,       
                        gdf_column* out_groupby_columns[],
                        gdf_column* out_aggregation_column,
                        gdf_context* ctxt)
{
  // Template the functor on the type of the aggregation column
  using op_type = op<aggregation_type>;

  bool group_by_keys_contain_nulls = false;
  std::vector<gdf_column*> orderby_cols_vect(num_groupby_cols);
  for (int i = 0; i < num_groupby_cols; i++){
    orderby_cols_vect[i] = in_groupby_columns[i];
    group_by_keys_contain_nulls = group_by_keys_contain_nulls || orderby_cols_vect[i]->null_count > 0;
  } 
  int32_t nrows = in_groupby_columns[0]->size;

  rmm::device_vector<int32_t> sorted_indices(nrows);
  gdf_column sorted_indices_col;
  gdf_error status = gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()), nullptr, nrows, GDF_INT32);
  
  if (status != GDF_SUCCESS)
    return status;
  
    
  auto clone = [](gdf_column *other) -> gdf_column *  {
    gdf_column *response = new gdf_column;
    char * data_dev = nullptr;
    gdf_valid_type * valid_dev = nullptr;

    int width;
    get_column_byte_width(other, &width);
    gdf_size_type allocated_size_data = width *  other->size;
    gdf_size_type allocated_size_valid =  gdf_valid_allocation_size(other->size); 
    
    //FIXME, add RMM_TRY, CUDA_TRY
    RMM_ALLOC((void**)&data_dev, allocated_size_data, 0);
    auto status = cudaMemcpy(data_dev, other->data, allocated_size_data, cudaMemcpyDeviceToDevice);
    if(status != cudaSuccess)
      throw std::runtime_error("cudaMemcpy error");
    
    RMM_ALLOC((void**)&valid_dev, allocated_size_valid, 0);
    if (other->null_count !=0 && other->valid != nullptr) {
      //FIXME, add RMM_TRY, CUDA_TRY
      status = cudaMemcpy(valid_dev, other->valid, allocated_size_valid, cudaMemcpyDeviceToDevice);
      if(status != cudaSuccess)
        throw std::runtime_error("cudaMemcpy error");
    }
    gdf_column_view_augmented(response, data_dev, valid_dev, other->size, other->dtype, other->null_count,  { TIME_UNIT_ms });
    return response;
  };


  // Cast the void* data to the appropriate type
  aggregation_type * in_agg_col = static_cast<aggregation_type *>(in_aggregation_column->data);
  // TODO Need to allow for the aggregation output type to be different from the aggregation input type
  aggregation_type * out_agg_col = static_cast<aggregation_type *>(out_aggregation_column->data);
  size_type output_size{0};
  gdf_error gdf_error_code = GDF_SUCCESS;

  gdf_column *in_agg_copy = nullptr;

  //FIXME: switch here!! with valid and without valids
  if (ctxt->flag_groupby_include_nulls == 1 || !group_by_keys_contain_nulls){  // SQL style wihtout NULLS
    auto is_count_distinct_op = typeid(op_type) == typeid(count_distinct_op<aggregation_type>);
    auto is_count_op =  typeid(op_type) == typeid(count_op<aggregation_type>);
    if (is_count_distinct_op) { // todo: check when the keys are the same as value
      in_agg_copy = clone(in_aggregation_column);
      orderby_cols_vect.push_back(in_agg_copy);  // NOTE: add another column, because for count_distinct we want to sort all including the agg_col
      auto in_groupby_columns_with_agg = orderby_cols_vect.data();

      // run order by and get new sort indexes (sort all columns including agg_in)
      status = gdf_order_by(in_groupby_columns_with_agg,             //input columns
                              nullptr,
                              num_groupby_cols + 1,            //number of columns in the first parameter (e.g. number of columsn to sort by)
                              &sorted_indices_col,             //a gdf_column that is pre allocated for storing sorted indices
                              ctxt);

       gdf_error_code = GroupbySortCountDistinct(num_groupby_cols,
                                  sorted_indices.data().get(), 
                                  in_groupby_columns,
                                  in_groupby_columns_with_agg,
                                  in_agg_col, 
                                  out_groupby_columns, 
                                  out_agg_col, 
                                  &output_size, 
                                  ctxt);
    } else if (is_count_op) {
      auto in_groupby_columns_with_agg = orderby_cols_vect.data();
      // run order by and get new sort indexes (sort all columns including agg_in)
      status = gdf_order_by(in_groupby_columns_with_agg,             //input columns
                              nullptr,
                              num_groupby_cols,            //number of columns in the first parameter (e.g. number of columsn to sort by)
                              &sorted_indices_col,             //a gdf_column that is pre allocated for storing sorted indices
                              ctxt);

       gdf_error_code = GroupbySortCount(num_groupby_cols,
                                  sorted_indices.data().get(), 
                                  in_groupby_columns,
                                  in_groupby_columns_with_agg,
                                  in_agg_col, 
                                  out_groupby_columns, 
                                  out_agg_col, 
                                  &output_size,
                                  ctxt); 
    } 
    else {
      // run order by and get new sort indexes
      status = gdf_order_by(&orderby_cols_vect[0],             //input columns
                              nullptr,
                              num_groupby_cols,                //number of columns in the first parameter (e.g. number of columsn to sort by)
                              &sorted_indices_col,             //a gdf_column that is pre allocated for storing sorted indices
                              ctxt);

      if (status != GDF_SUCCESS) return status;
      
      gdf_error_code = GroupbySort(num_groupby_cols,
                                    sorted_indices.data().get(), 
                                    in_groupby_columns, 
                                    in_agg_col, 
                                    out_groupby_columns, 
                                    out_agg_col, 
                                    &output_size, 
                                    op_type(),
                                    ctxt);
    }
  } else {
    // TODO< null version 
    if (typeid(op_type) == typeid(count_distinct_op<aggregation_type>)) { // todo: check when the keys are the same as value
      in_agg_copy = clone(in_aggregation_column);
      orderby_cols_vect.push_back(in_agg_copy);  // NOTE: add another column, because for count_distinct we want to sort all including the agg_col
      auto in_groupby_columns_with_agg = orderby_cols_vect.data();

      // run order by and get new sort indexes (sort all columns including agg_in)
      status = gdf_order_by(in_groupby_columns_with_agg,             //input columns
                              nullptr,
                              num_groupby_cols + 1,            //number of columns in the first parameter (e.g. number of columsn to sort by)
                              &sorted_indices_col,             //a gdf_column that is pre allocated for storing sorted indices
                              ctxt);

       gdf_error_code = GroupbySortCountDistinctWithNulls<aggregation_type>(
                                  num_groupby_cols,
                                  sorted_indices.data().get(), 
                                  in_groupby_columns,
                                  in_groupby_columns_with_agg,
                                  in_aggregation_column, 
                                  out_groupby_columns, 
                                  out_aggregation_column, 
                                  &output_size,
                                  ctxt);
    } else {
      // run order by and get new sort indexes
      status = gdf_order_by(&orderby_cols_vect[0],             //input columns
                              nullptr,
                              num_groupby_cols,                //number of columns in the first parameter (e.g. number of columsn to sort by)
                              &sorted_indices_col,             //a gdf_column that is pre allocated for storing sorted indices
                              ctxt);

      if (status != GDF_SUCCESS) return status;
      
      gdf_error_code = GroupbySortWithNulls<aggregation_type>(num_groupby_cols,
                                    sorted_indices.data().get(), 
                                    in_groupby_columns, 
                                    in_aggregation_column, 
                                    out_groupby_columns, 
                                    out_aggregation_column, 
                                    &output_size, 
                                    op_type(),
                                    ctxt);
    }
  }
  if(in_agg_copy != nullptr) {
    RMM_TRY(RMM_FREE(in_agg_copy->data, 0));
    if (in_agg_copy->valid)
        RMM_TRY(RMM_FREE(in_agg_copy->valid, 0));
  }
  out_aggregation_column->size = output_size; 

  return gdf_error_code;

}

template <template <typename> class, template<typename> class>
struct is_same_functor : std::false_type{};

template <template <typename> class T>
struct is_same_functor<T,T> : std::true_type{};

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Helper function for gdf_groupby_sort. Deduces the type of the aggregation
 * column and calls another function to perform the group by.
 * 
 */
/* ----------------------------------------------------------------------------*/
template <template <typename T> class op,
          typename size_type>
gdf_error dispatch_aggregation_type(size_type ncols,
                                    gdf_column* in_groupby_columns[],               
                                    gdf_column* in_aggregation_column,       
                                    gdf_column* out_groupby_columns[],
                                    gdf_column* out_aggregation_column,
                                    gdf_context* ctxt)
{


  gdf_dtype aggregation_column_type;

  // FIXME When the aggregation type is COUNT, use the type of the OUTPUT column
  // as the type of the aggregation column. This is required as there is a limitation 
  // Sort based groupby implementation where it's assumed the aggregation input column
  // and output column are the same type
  if(is_same_functor<count_op, op>::value || is_same_functor<count_distinct_op, op>::value)
  {
    aggregation_column_type = out_aggregation_column->dtype;
  }
  else
  {
    aggregation_column_type = in_aggregation_column->dtype;
  }

  // Deduce the type of the aggregation column and call function to perform GroupBy
  switch(aggregation_column_type)
  {
    case GDF_INT8:   
      { 
        return typed_groupby<int8_t, op>(ncols, in_groupby_columns, 
                                         in_aggregation_column, 
                                         out_groupby_columns, 
                                         out_aggregation_column, 
                                         ctxt);
      }
    case GDF_INT16:  
      { 
        return typed_groupby<int16_t, op>(ncols, in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    case GDF_INT32:  
      { 
        return typed_groupby<int32_t, op>(ncols,
                                          in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    case GDF_INT64:  
      { 
        return typed_groupby<int64_t, op>(ncols, in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    case GDF_FLOAT32:
      { 
        return typed_groupby<float, op>(ncols, in_groupby_columns, 
                                        in_aggregation_column, 
                                        out_groupby_columns, 
                                        out_aggregation_column, 
                                        ctxt);
      }
    case GDF_FLOAT64:
      { 
        return typed_groupby<double, op>(ncols, in_groupby_columns, 
                                         in_aggregation_column, 
                                         out_groupby_columns, 
                                         out_aggregation_column, 
                                         ctxt);
      }
    case GDF_DATE32:    
      {
        return typed_groupby<int32_t, op>(ncols, in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    case GDF_DATE64:   
      {
        return typed_groupby<int64_t, op>(ncols, in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    case GDF_TIMESTAMP:
      {
        return typed_groupby<int64_t, op>(ncols, in_groupby_columns, 
                                          in_aggregation_column, 
                                          out_groupby_columns, 
                                          out_aggregation_column, 
                                          ctxt);
      }
    default:
      std::cerr << "Unsupported aggregation column type: " << aggregation_column_type << std::endl;
      return GDF_UNSUPPORTED_DTYPE;
  }
}


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This function provides the libgdf entry point for a Sort-based group-by.
 * Performs a Group-By operation on an arbitrary number of columns with a single aggregation column.
 * 
 * @Param[in] ncols The number of columns to group-by
 * @Param[in] in_groupby_columns[] The columns to group-by
 * @Param[in,out] in_aggregation_column The column to perform the aggregation on
 * @Param[in,out] out_groupby_columns[] A preallocated buffer to store the resultant group-by columns
 * @Param[in,out] out_aggregation_column A preallocated buffer to store the resultant aggregation column
 * @tparam[in] aggregation_operation A functor that defines the aggregation operation
 * 
 * @Returns gdf_error
 */
/* ----------------------------------------------------------------------------*/
template <template <typename aggregation_type> class aggregation_operation,
          typename size_type>
gdf_error gdf_group_by_sort(size_type ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column,
                            gdf_context* ctxt)
{


  // Make sure the inputs are not null
  if( (0 == ncols) 
      || (nullptr == in_groupby_columns) 
      || (nullptr == in_aggregation_column))
  {
    return GDF_DATASET_EMPTY;
  }

  // Make sure the output buffers have already been allocated
  if( (nullptr == out_groupby_columns) 
      || (nullptr == out_aggregation_column))
  {
    return GDF_DATASET_EMPTY;
  }

  // If there are no rows in the input, return successfully
  if ((0 == in_groupby_columns[0]->size) 
      || (0 == in_aggregation_column->size) )
  {
    return GDF_SUCCESS;
  }
  //@alex TODO remove gdf_table references
  // Wrap the groupby input and output columns in a gdf_table
  // std::unique_ptr< const gdf_table<size_type> > ncols, in_groupby_columns{new gdf_table<size_type>(ncols, in_groupby_columns)};
  // std::unique_ptr< gdf_table<size_type> > out_groupby_columns{new gdf_table<size_type>(ncols, out_groupby_columns)};

  return dispatch_aggregation_type<aggregation_operation>(ncols,
                                                          in_groupby_columns, 
                                                          in_aggregation_column, 
                                                          out_groupby_columns, 
                                                          out_aggregation_column, 
                                                          ctxt);
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Creates a gdf_column of a specified size and data type
 * 
 * @Param size The number of elements in the gdf_column
 * @tparam col_type The datatype of the gdf_column
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
template<typename col_type>
gdf_column create_gdf_column(const size_t size)
{
  gdf_column the_column;

  // Deduce the type and set the gdf_dtype accordingly
  gdf_dtype gdf_col_type;
  if(std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
  else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;
  else assert(false && "Invalid type passed to create_gdf_column");

  // Fill the gdf_column struct
  the_column.size = size;
  the_column.dtype = gdf_col_type;
  the_column.valid = nullptr;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column.dtype_info = extra_info;

  // Allocate the buffer for the column
  // TODO error checking?
  RMM_ALLOC((void**)&the_column.data, the_column.size * sizeof(col_type), 0); // TODO: non-default stream?
  
  return the_column;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Given a column for the SUM and COUNT aggregations, computes the AVG
 * aggregation column result as AVG[i] = SUM[i] / COUNT[i].
 * 
 * @Param[out] avg_column The output AVG aggregation column
 * @Param count_column The input COUNT aggregation column
 * @Param sum_column The input SUM aggregation column
 * @tparam sum_type The type used for the SUM column
 * @tparam avg_type The type used for the AVG column
 */
/* ----------------------------------------------------------------------------*/
template <typename sum_type, typename avg_type>
void compute_average(gdf_column * avg_column, gdf_column const & count_column, gdf_column const & sum_column)
{
  const size_t output_size = count_column.size;

  // Wrap raw device pointers in thrust device ptrs to enable usage of thrust::transform
  thrust::device_ptr<sum_type> d_sums = thrust::device_pointer_cast(static_cast<sum_type*>(sum_column.data));
  thrust::device_ptr<size_t> d_counts = thrust::device_pointer_cast(static_cast<size_t*>(count_column.data));
  thrust::device_ptr<avg_type> d_avg  = thrust::device_pointer_cast(static_cast<avg_type*>(avg_column->data));

  auto average_op =  [] __device__ (sum_type sum, size_t count)->avg_type { return (sum / static_cast<avg_type>(count)); };

  // Computes the average into the passed in output buffer for the average column
  thrust::transform(rmm::exec_policy()->on(0), d_sums, d_sums + output_size, d_counts, d_avg, average_op);

  // Update the size of the average column
  avg_column->size = output_size;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Computes the SUM and COUNT aggregations for the group by inputs. Calls 
 * another function to compute the AVG aggregator based on the SUM and COUNT results.
 * 
 * @Param ncols The number of input groupby columns
 * @Param in_groupby_columns[] The groupby input columns
 * @Param in_aggregation_column The aggregation input column
 * @Param out_groupby_columns[] The output groupby columns
 * @Param out_aggregation_column The output aggregation column
 * @tparam sum_type The type used for the SUM aggregation output column
 * 
 * @Returns gdf_error with error code on failure, otherwise GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
template <typename sum_type>
gdf_error multi_pass_avg(int ncols,               
                         gdf_column* in_groupby_columns[],        
                         gdf_column* in_aggregation_column,       
                         gdf_column* out_groupby_columns[],
                         gdf_column* out_aggregation_column)
{
  // Allocate intermediate output gdf_columns for the output of the Count and Sum aggregations
  const size_t output_size = out_aggregation_column->size;

  // Make sure the result is sorted so the output is in identical order
  gdf_context ctxt;
  ctxt.flag_distinct = false;
  ctxt.flag_method = GDF_SORT;
  ctxt.flag_sort_result = 1;
  
  // FIXME Currently, Sort based groupby assumes the type of the input aggregation column and 
  // the output aggregation column are the same. This doesn't work for COUNT

  // // Compute the counts for each key 
  gdf_column count_output = create_gdf_column<size_t>(output_size);
  gdf_group_by_sort<count_op>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, &count_output, &ctxt);

  // // Compute the sum for each key. Should be okay to reuse the groupby column output
  gdf_column sum_output = create_gdf_column<sum_type>(output_size);
  gdf_group_by_sort<sum_op>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, &sum_output, &ctxt); 

  // // Compute the average from the Sum and Count columns and store into the passed in aggregation output buffer
  const gdf_dtype gdf_output_type = out_aggregation_column->dtype;
  switch(gdf_output_type){
    case GDF_INT8:    { compute_average<sum_type, int8_t>( out_aggregation_column, count_output, sum_output); break; }
    case GDF_INT16:   { compute_average<sum_type, int16_t>( out_aggregation_column, count_output, sum_output); break; }
    case GDF_INT32:   { compute_average<sum_type, int32_t>( out_aggregation_column, count_output, sum_output); break; }
    case GDF_INT64:   { compute_average<sum_type, int64_t>( out_aggregation_column, count_output, sum_output); break; }
    case GDF_FLOAT32: { compute_average<sum_type, float>( out_aggregation_column, count_output, sum_output); break; }
    case GDF_FLOAT64: { compute_average<sum_type, double>( out_aggregation_column, count_output, sum_output); break; }
    default: return GDF_UNSUPPORTED_DTYPE;
  }

  // Free intermediate storage
  RMM_TRY( RMM_FREE(count_output.data, 0) );
  RMM_TRY( RMM_FREE(sum_output.data, 0) );
  
  return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the AVG aggregation for a Sort-based group by. This aggregator
 * requires its own function as AVG is implemented via the COUNT and SUM aggregators.
 * 
 * @Param ncols The number of columns to groupby
 * @Param in_groupby_columns[] The input groupby columns
 * @Param in_aggregation_column The input aggregation column
 * @Param out_groupby_columns[] The output groupby columns
 * @Param out_aggregation_column The output aggregation column
 * 
 * @Returns gdf_error with error code on failure, otherwise GDF_SUCESS
 */
/* ----------------------------------------------------------------------------*/
static inline gdf_error gdf_group_by_sort_avg(int ncols,               
                                gdf_column* in_groupby_columns[],        
                                gdf_column* in_aggregation_column,       
                                gdf_column* out_groupby_columns[],
                                gdf_column* out_aggregation_column)
{
  // Deduce the type used for the SUM aggregation, assuming we use the same type as the aggregation column
  const gdf_dtype gdf_sum_type = in_aggregation_column->dtype;
  switch(gdf_sum_type){
    case GDF_INT8:   { return multi_pass_avg<int8_t>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    case GDF_INT16:  { return multi_pass_avg<int16_t>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    case GDF_INT32:  { return multi_pass_avg<int32_t>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    case GDF_INT64:  { return multi_pass_avg<int64_t>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    case GDF_FLOAT32:{ return multi_pass_avg<float>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    case GDF_FLOAT64:{ return multi_pass_avg<double>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);}
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

} //END: namespace group_by_sort 
#endif
