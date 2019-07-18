#ifndef SORT_GROUPBY_SORT_H
#define SORT_GROUPBY_SORT_H
/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Alexander Ocsa <alexander@blazingdb.com>
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

#include "cudf/cudf.h"
#include "utilities/error_utils.hpp"

#include "groupby/aggregation_operations.hpp"

#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"

namespace cudf{
namespace groupby{
namespace sort{
namespace detail {     

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Creates a gdf_column of a specified siz~e and data type
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
  // // Allocate intermediate output gdf_columns for the output of the Count and Sum aggregations
  // const size_t output_size = out_aggregation_column->size;

  // // Make sure the result is sorted so the output is in identical order
  // gdf_context ctxt;
  // ctxt.flag_distinct = false;
  // ctxt.flag_method = GDF_SORT;
  // ctxt.flag_sort_result = 1;
  
  // // FIXME Currently, Sort based groupby assumes the type of the input aggregation column and 
  // // the output aggregation column are the same. This doesn't work for COUNT

  // // // Compute the counts for each key 
  // gdf_column count_output = create_gdf_column<size_t>(output_size);
  // gdf_agg_op count_op[] = {GDF_COUNT};
  // gdf_column* out_agg_count[] =  {&count_output};
  // // gdf_group_by_sort(in_groupby_columns, ncols, &in_aggregation_column, 1, count_op, out_groupby_columns, out_agg_count, &ctxt);

  // // // Compute the sum for each key. Should be okay to reuse the groupby column output
  // gdf_column sum_output = create_gdf_column<sum_type>(output_size);
  // gdf_agg_op sum_op[] = {GDF_SUM};
  // gdf_column* out_agg_sum[] =  {&sum_output};
  // // gdf_group_by_sort(in_groupby_columns, ncols, &in_aggregation_column, 1, sum_op, out_groupby_columns, out_agg_sum, &ctxt);

  // // // Compute the average from the Sum and Count columns and store into the passed in aggregation output buffer
  // const gdf_dtype gdf_output_type = out_aggregation_column->dtype;
  // switch(gdf_output_type){
  //   case GDF_INT8:    { compute_average<sum_type, int8_t>( out_aggregation_column, count_output, sum_output); break; }
  //   case GDF_INT16:   { compute_average<sum_type, int16_t>( out_aggregation_column, count_output, sum_output); break; }
  //   case GDF_INT32:   { compute_average<sum_type, int32_t>( out_aggregation_column, count_output, sum_output); break; }
  //   case GDF_INT64:   { compute_average<sum_type, int64_t>( out_aggregation_column, count_output, sum_output); break; }
  //   case GDF_FLOAT32: { compute_average<sum_type, float>( out_aggregation_column, count_output, sum_output); break; }
  //   case GDF_FLOAT64: { compute_average<sum_type, double>( out_aggregation_column, count_output, sum_output); break; }
  //   default: return GDF_UNSUPPORTED_DTYPE;
  // }

  // // Free intermediate storage
  // RMM_TRY( RMM_FREE(count_output.data, 0) );
  // RMM_TRY( RMM_FREE(sum_output.data, 0) );
  
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

} //END: namespace detail
} //END: namespace sort
} //END: namespace groupby
} //END: namespace cudf 
#endif
