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

#ifndef GROUPBY_SORT_COMPUTE_API_H
#define GROUPBY_SORT_COMPUTE_API_H

#include "cudf.h"
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

#include "rmm/thrust_rmm_allocator.h"
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
#include "utilities/bit_util.cuh"


/* --------------------------------------------------------------------------*/
/** 
* @Synopsis Performs the groupby operation for an arbtirary number of groupby columns and
* and a single aggregation column.
* 
* @Param[in] groupby_input_table The set of columns to groupby
* @Param[in] in_aggregation_column The column to perform the aggregation on. These act as the hash table values
* @Param[out] groupby_output_table Preallocated buffer(s) for the groupby column result. This will hold a single
* entry for every unique row in the input table.
* @Param[out] out_aggregation_column Preallocated output buffer for the resultant aggregation column that 
*                                     corresponds to the out_groupby_column where entry 'i' is the aggregation 
*                                     for the group out_groupby_column[i] 
* @Param out_size The size of the output
* @Param aggregation_op The aggregation operation to perform 
* @Param sort_result Flag to optionally sort the output table
* 
* @Returns   
*/
/* ----------------------------------------------------------------------------*/
template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
gdf_error GroupbySort(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],       
                        const aggregation_type * const in_aggregation_column,
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols, &(in_groupby_columns[0]));
  auto comp = row_equality_comparator<false>(*device_input_table, true);
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto agg_col_iter = thrust::make_permutation_iterator(in_aggregation_column, d_sorted_indices);

  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              agg_col_iter,  
                              d_sorted_indices, 
                              out_aggregation_column,  
                              comp,
                              aggregation_op);
  size_type new_size = thrust::distance(out_aggregation_column, ret.second);
  *out_size = new_size;

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);
  
	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}

  return GDF_SUCCESS;
}


struct GdfValidToBool {
  gdf_valid_type* d_valid;

  __device__
  bool operator () (gdf_size_type idx)
  {
      return gdf_is_valid(d_valid, idx);
  }
};

struct GdfBoolToValid {
  gdf_valid_type* d_valid;
  bool* d_bools;

  __device__
  void operator () (gdf_size_type idx)
  {
    if (d_bools[idx])
      gdf::util::turn_bit_on(d_valid, idx);
    else
      gdf::util::turn_bit_off(d_valid, idx);
  }
};

rmm::device_vector<bool> get_bools_from_gdf_valid(gdf_column* column) {
  rmm::device_vector<bool> d_bools(column->size);
  thrust::transform(thrust::make_counting_iterator(static_cast<gdf_size_type>(0)), thrust::make_counting_iterator(column->size), d_bools.begin(), GdfValidToBool{column->valid});
  return d_bools;
}

void set_bools_for_gdf_valid(gdf_column* column, rmm::device_vector<bool>& d_bools) {
  thrust::for_each(thrust::make_counting_iterator(static_cast<gdf_size_type>(0)), thrust::make_counting_iterator(column->size), GdfBoolToValid{column->valid, d_bools.data().get()});
}

template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
gdf_error GroupbySortWithNulls(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],       
                        gdf_column* in_aggregation_column,
                        gdf_column* out_groupby_columns[],
                        gdf_column* out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        gdf_context* ctxt)
{

  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols, &(in_groupby_columns[0]));
  auto comp = row_equality_comparator<true>(*device_input_table, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);
  rmm::device_vector<bool> d_in_agg_col_valids = get_bools_from_gdf_valid(in_aggregation_column);

  auto agg_col_iter = thrust::make_permutation_iterator( (aggregation_type*)in_aggregation_column->data, d_sorted_indices);
	auto agg_col_valid_iter = thrust::make_permutation_iterator(d_in_agg_col_valids.begin(), d_sorted_indices); 
	auto agg_col_zip_iter = thrust::make_zip_iterator(thrust::make_tuple(agg_col_iter, agg_col_valid_iter));

  rmm::device_vector<bool> d_out_agg_col_valids = get_bools_from_gdf_valid(out_aggregation_column);
	auto out_agg_col_zip_iter = thrust::make_zip_iterator( thrust::make_tuple((aggregation_type*)out_aggregation_column->data, d_out_agg_col_valids.begin()));

  using op_with_valids = typename aggregation_operation::with_valids;
  op_with_valids agg_op;
  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, // input keys
                              agg_col_zip_iter,                             // input values
                              d_sorted_indices,                         // output keys
                              out_agg_col_zip_iter,  // output values
                              comp,
                              agg_op);
  auto iter_tuple = ret.second.get_iterator_tuple();

  size_type new_size = thrust::distance((aggregation_type*)out_aggregation_column->data, thrust::get<0>(iter_tuple));

  *out_size = new_size;

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);
  
	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}
  out_aggregation_column->size = new_size;
  set_bools_for_gdf_valid(out_aggregation_column, d_out_agg_col_valids);

  return GDF_SUCCESS;
}


template< typename aggregation_type,
          typename size_type>
gdf_error GroupbySortCountDistinct(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_groupby_columns_with_agg[],       
                        const aggregation_type * const in_aggregation_column, //not used
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols + 1, &(in_groupby_columns_with_agg[0]));
  auto comp = row_equality_comparator<false>(*device_input_table, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<aggregation_type> out_col(nrows);
  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              thrust::make_constant_iterator<int32_t>(1),  
                              d_sorted_indices, 
                              out_col.data().get(),  
                              comp
                              );
  size_type new_size = thrust::distance(out_col.data().get(), ret.second);
  *out_size = new_size;

  CUDA_TRY(cudaMemcpy(out_aggregation_column, out_col.data().get(), new_size * sizeof(aggregation_type), cudaMemcpyDeviceToDevice));

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}

  return GDF_SUCCESS;
}


template< typename aggregation_type,
          typename size_type>
gdf_error GroupbySortCount(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_groupby_columns_with_agg[],       
                        const aggregation_type * const in_aggregation_column, //not used
                        gdf_column* out_groupby_columns[],
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols, &(in_groupby_columns_with_agg[0]));
  auto comp = row_equality_comparator<false>(*device_input_table, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              thrust::make_constant_iterator<int32_t>(1),  
                              d_sorted_indices, 
                              (aggregation_type*)out_aggregation_column,  
                              comp
                              );
  size_type new_size = thrust::distance((aggregation_type*)out_aggregation_column, ret.second);
  *out_size = new_size;

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}
  return GDF_SUCCESS;
}


template< typename aggregation_type,
          typename size_type>
gdf_error GroupbySortCountDistinctWithNulls(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_groupby_columns_with_agg[],       
                        gdf_column* in_aggregation_column,//not used
                        gdf_column* out_groupby_columns[],
                        gdf_column* out_aggregation_column,
                        size_type * out_size,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols + 1, &(in_groupby_columns_with_agg[0]));
  auto comp = row_equality_comparator<true>(*device_input_table, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<bool> d_in_agg_col_valids = get_bools_from_gdf_valid(in_aggregation_column);

  auto agg_col_iter = thrust::make_constant_iterator<aggregation_type>(1);
	auto agg_col_valid_iter = thrust::make_permutation_iterator(d_in_agg_col_valids.begin(), d_sorted_indices); 
	auto agg_col_zip_iter = thrust::make_zip_iterator(thrust::make_tuple(agg_col_iter, agg_col_valid_iter));

  rmm::device_vector<bool> d_out_agg_col_valids = get_bools_from_gdf_valid(out_aggregation_column);
  rmm::device_vector<aggregation_type> out_col(nrows);

	auto out_agg_col_zip_iter = thrust::make_zip_iterator( thrust::make_tuple((aggregation_type*)out_col.data().get(), d_out_agg_col_valids.begin()));

  count_distinct_op_valids<aggregation_type> agg_op;
  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              agg_col_zip_iter,  
                              d_sorted_indices, 
                              out_agg_col_zip_iter,  
                              comp,
                              agg_op);
  auto iter_tuple = ret.second.get_iterator_tuple();
  size_type new_size = thrust::distance((aggregation_type*)out_col.data().get(), thrust::get<0>(iter_tuple));

  *out_size = new_size;

  CUDA_TRY(cudaMemcpy(out_aggregation_column->data, out_col.data().get(), new_size * sizeof(aggregation_type), cudaMemcpyDeviceToDevice));//lesson final, not reuse out_aggregation_column 

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}
  out_aggregation_column->size = new_size;
  set_bools_for_gdf_valid(out_aggregation_column, d_out_agg_col_valids);

  return GDF_SUCCESS;
}


template< typename aggregation_type,
          typename size_type>
gdf_error GroupbySortCountWithNulls(size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_groupby_columns_with_agg[],       
                        gdf_column* in_aggregation_column,//not used
                        gdf_column* out_groupby_columns[],
                        gdf_column* out_aggregation_column,
                        size_type * out_size,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols, &(in_groupby_columns_with_agg[0]));
  auto comp = row_equality_comparator<true>(*device_input_table, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  rmm::device_vector<bool> d_in_agg_col_valids = get_bools_from_gdf_valid(in_aggregation_column);

  auto agg_col_iter = thrust::make_constant_iterator<aggregation_type>(1);
	auto agg_col_valid_iter = thrust::make_permutation_iterator(d_in_agg_col_valids.begin(), d_sorted_indices); 
	auto agg_col_zip_iter = thrust::make_zip_iterator(thrust::make_tuple(agg_col_iter, agg_col_valid_iter));

  rmm::device_vector<bool> d_out_agg_col_valids = get_bools_from_gdf_valid(out_aggregation_column);

	auto out_agg_col_zip_iter = thrust::make_zip_iterator( thrust::make_tuple((aggregation_type*)out_aggregation_column->data, d_out_agg_col_valids.begin()));

  count_op_valids<aggregation_type> agg_op;
  auto ret =
        thrust::reduce_by_key(exec,
                              d_sorted_indices, d_sorted_indices+nrows, 
                              agg_col_zip_iter,  
                              d_sorted_indices, 
                              out_agg_col_zip_iter,  
                              comp,
                              agg_op);
  auto iter_tuple = ret.second.get_iterator_tuple();
  size_type new_size = thrust::distance((aggregation_type*)out_aggregation_column->data, thrust::get<0>(iter_tuple));

  *out_size = new_size;

  // run gather operation to establish new order
  cudf::table table_in(in_groupby_columns, num_groupby_cols);
  cudf::table table_out(out_groupby_columns, num_groupby_cols);
  
  cudf::gather(&table_in, d_sorted_indices, &table_out);

	for (int i = 0; i < num_groupby_cols; i++) {
		out_groupby_columns[i]->size = new_size;
	}
  out_aggregation_column->size = new_size;
  set_bools_for_gdf_valid(out_aggregation_column, d_out_agg_col_valids);

  return GDF_SUCCESS;
}


#endif
