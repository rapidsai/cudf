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

#include "groupby_count_valid.h"
#include "rmm/thrust_rmm_allocator.h"

#include <cuda_runtime.h>

#include "cudf/copying.hpp"
#include "cudf/cudf.h"
#include "utilities/error_utils.hpp"

#include "groupby/aggregation_operations.hpp"

#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"

#include <limits>
#include <memory>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

#include "groupby_valid_helpers.h"

namespace {
 
template< typename aggregation_type>
gdf_error GroupbySortCountWithNulls(gdf_size_type num_groupby_cols,
                        int32_t* d_sorted_indices,
                        gdf_column* in_groupby_columns[],
                        gdf_column* in_aggregation_column,//not used
                        gdf_column* out_groupby_columns[],
                        gdf_column* out_aggregation_column,
                        gdf_size_type * out_size,
                        gdf_context* ctxt)
{
  int32_t nrows = in_groupby_columns[0]->size;
  
  auto device_input_table = device_table::create(num_groupby_cols, &(in_groupby_columns[0]));
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
  gdf_size_type new_size = thrust::distance((aggregation_type*)out_aggregation_column->data, thrust::get<0>(iter_tuple));

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

template <typename aggregation_type>
gdf_error typed_groupby_sort(gdf_size_type num_groupby_cols,
                             gdf_column *in_groupby_columns[],
                             gdf_column *in_aggregation_column,
                             gdf_column *out_groupby_columns[],
                             gdf_column *out_aggregation_column,
                             gdf_context *ctxt,
                             rmm::device_vector<int32_t> &sorted_indices) {
  gdf_error gdf_error_code = GDF_SUCCESS;
  gdf_size_type output_size{0}; 

  gdf_error_code = GroupbySortCountWithNulls<aggregation_type>(
      num_groupby_cols, sorted_indices.data().get(), in_groupby_columns,
      in_aggregation_column, out_groupby_columns, out_aggregation_column,
      &output_size, ctxt);
  out_aggregation_column->size = output_size;
  return gdf_error_code; 
}

struct dispatch_groupby_forwarder {
  template <typename TypeAgg, typename... Ts>
  gdf_error operator()(Ts &&... args) {
    return typed_groupby_sort<TypeAgg>(std::forward<Ts>(args)...);
  }
};

} // namespace

gdf_error gdf_group_by_count_with_valids(
    gdf_size_type ncols, gdf_column *in_groupby_columns[],
    gdf_column *in_aggregation_column, gdf_column *out_groupby_columns[],
    gdf_column *out_aggregation_column, gdf_agg_op agg_op, gdf_context *ctxt,
    rmm::device_vector<int32_t> &sorted_indices) {

  // Make sure the inputs are not null
  if ((0 == ncols) || (nullptr == in_groupby_columns) ||
      (nullptr == in_aggregation_column)) {
    return GDF_DATASET_EMPTY;
  }

  // Make sure the output buffers have already been allocated
  if ((nullptr == out_groupby_columns) || (nullptr == out_aggregation_column)) {
    return GDF_DATASET_EMPTY;
  }

  // If there are no rows in the input, return successfully
  if ((0 == in_groupby_columns[0]->size) ||
      (0 == in_aggregation_column->size)) {
    return GDF_SUCCESS;
  }
  
  gdf_error gdf_error_code{GDF_SUCCESS}; 
  auto  aggregation_column_type = out_aggregation_column->dtype;
  gdf_error_code = groupby_type_dispatcher(
      aggregation_column_type, dispatch_groupby_forwarder{},
      ncols, in_groupby_columns, in_aggregation_column, out_groupby_columns,
      out_aggregation_column, ctxt, sorted_indices);
  return gdf_error_code;
}