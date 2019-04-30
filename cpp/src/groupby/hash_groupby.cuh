#ifndef HASH_GROUPBY_H
#define HASH_GROUPBY_H
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
#include <utilities/error_utils.hpp>
#include <table/device_table.cuh>
#include <types.hpp>
#include <cudf.h>

#include "groupby_compute_api.h"
#include "aggregation_operations.hpp"

#include <type_traits>

/* --------------------------------------------------------------------------*/
/** 
 * @brief Calls the Hash Based group by compute API to compute the groupby with 
 * aggregation.
 * 
 * @param input_keys The input groupby table
 * @param in_aggregation_column The input aggregation column
 * @param output_keys The output groupby table
 * @param out_aggregation_column The output aggregation column
 * @param sort_result Flag to optionally sort the output
 * @tparam aggregation_type  The type of the aggregation column
 * @tparam op A binary functor that implements the aggregation operation
 * 
 * @returns On failure, returns appropriate error code. Otherwise, GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
template <typename aggregation_type, 
          template <typename T> class op>
gdf_error typed_groupby(cudf::table const & input_keys,
                        gdf_column* in_aggregation_column,       
                        cudf::table & output_keys,
                        gdf_column* out_aggregation_column,
                        bool sort_result = false)
{
  // Template the functor on the type of the aggregation column
  using op_type = op<aggregation_type>;

  // Cast the void* data to the appropriate type
  aggregation_type * in_agg_col = static_cast<aggregation_type *>(in_aggregation_column->data);
  // TODO Need to allow for the aggregation output type to be different from the aggregation input type
  aggregation_type * out_agg_col = static_cast<aggregation_type *>(out_aggregation_column->data);

  gdf_size_type output_size{0};

  gdf_error gdf_error_code = GroupbyHash(input_keys, 
                                         in_agg_col, 
                                         output_keys, 
                                         out_agg_col, 
                                         &output_size, 
                                         op_type(), 
                                         sort_result);

  out_aggregation_column->size = output_size;

  return gdf_error_code;

}

template <template <typename> class, template<typename> class>
struct is_same_functor : std::false_type{};

template <template <typename> class T>
struct is_same_functor<T,T> : std::true_type{};

/**---------------------------------------------------------------------------*
 * @brief Functor for `typed_groupby` to use with `type_dispatcher`
 * 
 *---------------------------------------------------------------------------**/
template <template <typename T> class op>
struct typed_groupby_functor
{
  template <typename TypeAgg, typename... Ts>
  gdf_error operator()(Ts&&... args)
  {
    return typed_groupby<TypeAgg, op>(std::forward<Ts>(args)...);    
  }
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Helper function for gdf_groupby_hash. Deduces the type of the aggregation
 * column and calls another function to perform the group by.
 * 
 */
/* ----------------------------------------------------------------------------*/
template <template <typename T> class op>
gdf_error dispatch_aggregation_type(cudf::table const & input_keys,        
                                    gdf_column* in_aggregation_column,       
                                    cudf::table & output_keys,
                                    gdf_column* out_aggregation_column,
                                    bool sort_result = false)
{


  gdf_dtype aggregation_column_type;

  // FIXME When the aggregation type is COUNT, use the type of the OUTPUT column
  // as the type of the aggregation column. This is required as there is a limitation 
  // hash based groupby implementation where it's assumed the aggregation input column
  // and output column are the same type
  if(is_same_functor<count_op, op>::value)
  {
    aggregation_column_type = out_aggregation_column->dtype;
  }
  else
  {
    aggregation_column_type = in_aggregation_column->dtype;
  }

  return cudf::type_dispatcher(aggregation_column_type,
                              typed_groupby_functor<op>{},
                              input_keys, 
                              in_aggregation_column, 
                              output_keys, 
                              out_aggregation_column, 
                              sort_result);

}


/* --------------------------------------------------------------------------*/
/** 
 * @brief  This function provides the libgdf entry point for a hash-based group-by.
 * Performs a Group-By operation on an arbitrary number of columns with a single aggregation column.
 * 
 * @param[in] ncols The number of columns to group-by
 * @param[in] in_groupby_columns[] The columns to group-by
 * @param[in,out] in_aggregation_column The column to perform the aggregation on
 * @param[in,out] out_groupby_columns[] A preallocated buffer to store the resultant group-by columns
 * @param[in,out] out_aggregation_column A preallocated buffer to store the resultant aggregation column
 * @tparam[in] aggregation_operation A functor that defines the aggregation operation
 * 
 * @returns gdf_error
 */
/* ----------------------------------------------------------------------------*/
template <template <typename aggregation_type> class aggregation_operation>
gdf_error gdf_group_by_hash(gdf_size_type ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column,
                            bool sort_result = false)
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

  cudf::table input_keys{in_groupby_columns, ncols};
  cudf::table output_keys{out_groupby_columns, ncols};

  return dispatch_aggregation_type<aggregation_operation>(input_keys, 
                                                          in_aggregation_column, 
                                                          output_keys, 
                                                          out_aggregation_column, 
                                                          sort_result);
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Creates a gdf_column of a specified size and data type
 * 
 * @param size The number of elements in the gdf_column
 * @tparam col_type The datatype of the gdf_column
 * 
 * @returns   
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
 * @brief Given a column for the SUM and COUNT aggregations, computes the AVG
 * aggregation column result as AVG[i] = SUM[i] / COUNT[i].
 * 
 * @param[out] avg_column The output AVG aggregation column
 * @param count_column The input COUNT aggregation column
 * @param sum_column The input SUM aggregation column
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

/**---------------------------------------------------------------------------*
 * @brief Functor for `compute_average` to be used with `type_dispatcher`
 * 
 *---------------------------------------------------------------------------**/
template <typename sum_type>
struct compute_average_functor {
  template <typename T, typename... Ts>
  typename std::enable_if_t<std::is_arithmetic<T>::value, gdf_error>
  operator()(Ts&&... args)
  {
    compute_average<sum_type, T>(std::forward<Ts>(args)...);
    return GDF_SUCCESS;
  }

  template <typename T, typename... Ts>
  typename std::enable_if_t<!std::is_arithmetic<T>::value, gdf_error>
  operator()(Ts&&... args)
  {
    return GDF_UNSUPPORTED_DTYPE;
  }
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes the SUM and COUNT aggregations for the group by inputs. Calls 
 * another function to compute the AVG aggregator based on the SUM and COUNT results.
 * 
 * @param ncols The number of input groupby columns
 * @param in_groupby_columns[] The groupby input columns
 * @param in_aggregation_column The aggregation input column
 * @param out_groupby_columns[] The output groupby columns
 * @param out_aggregation_column The output aggregation column
 * @tparam sum_type The type used for the SUM aggregation output column
 * 
 * @returns gdf_error with error code on failure, otherwise GDF_SUCCESS
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
  bool sort_result = true;

  // FIXME Currently, hash based groupby assumes the type of the input aggregation column and 
  // the output aggregation column are the same. This doesn't work for COUNT

  // Compute the counts for each key 
  gdf_column count_output = create_gdf_column<size_t>(output_size);
  gdf_group_by_hash<count_op>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, &count_output, sort_result);

  // Compute the sum for each key. Should be okay to reuse the groupby column output
  gdf_column sum_output = create_gdf_column<sum_type>(output_size);
  gdf_group_by_hash<sum_op>(ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, &sum_output, sort_result); 

  // Compute the average from the Sum and Count columns and store into the passed in aggregation output buffer
  auto result = cudf::type_dispatcher(out_aggregation_column->dtype,
                                      compute_average_functor<sum_type>{},
                                      out_aggregation_column, count_output, sum_output);

  // Free intermediate storage
  RMM_TRY( RMM_FREE(count_output.data, 0) );
  RMM_TRY( RMM_FREE(sum_output.data, 0) );
  
  return result;
}

/**---------------------------------------------------------------------------*
 * @brief Functor for `multi_pass_avg` to be used with `type_dispatcher`
 * 
 *---------------------------------------------------------------------------**/
struct multi_pass_avg_functor {
  template <typename T, typename... Ts>
  typename std::enable_if_t<std::is_arithmetic<T>::value, gdf_error>
  operator()(Ts&&... args)
  {
    return multi_pass_avg<T>(std::forward<Ts>(args)...);
  }

  template <typename T, typename... Ts>
  typename std::enable_if_t<!std::is_arithmetic<T>::value, gdf_error>
  operator()(Ts&&... args)
  {
    return GDF_UNSUPPORTED_DTYPE;
  }
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the AVG aggregation for a hash-based group by. This aggregator
 * requires its own function as AVG is implemented via the COUNT and SUM aggregators.
 * 
 * @param ncols The number of columns to groupby
 * @param in_groupby_columns[] The input groupby columns
 * @param in_aggregation_column The input aggregation column
 * @param out_groupby_columns[] The output groupby columns
 * @param out_aggregation_column The output aggregation column
 * 
 * @returns gdf_error with error code on failure, otherwise GDF_SUCESS
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_group_by_hash_avg(int ncols,               
                                gdf_column* in_groupby_columns[],        
                                gdf_column* in_aggregation_column,       
                                gdf_column* out_groupby_columns[],
                                gdf_column* out_aggregation_column)
{
  // Deduce the type used for the SUM aggregation, assuming we use the same type as the aggregation column

  return cudf::type_dispatcher(in_aggregation_column->dtype,
                              multi_pass_avg_functor{},
                              ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns, out_aggregation_column);
}

#endif
