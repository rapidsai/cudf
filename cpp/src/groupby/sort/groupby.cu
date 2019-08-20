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

#include <algorithm>
#include <cassert>
#include <thrust/fill.h>
#include <tuple>


#include <cudf/cudf.h>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
 
 #include "../common/aggregation_requests.hpp"
#include "../common/util.hpp"
#include "groupby.hpp"
#include "groupby_kernels.cuh"  

using namespace cudf::groupby::common;

namespace cudf {
namespace groupby {
namespace sort {

namespace {   

cudf::table compose_inputs(cudf::table input_table, gdf_column* col) {
  std::vector<gdf_column*> output(input_table.num_columns());  
  std::transform(input_table.begin(), input_table.end(), output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  output.push_back(col);

  gdf_column **group_by_input_key = output.data();
  return cudf::table{group_by_input_key, input_table.num_columns() + 1};
}

cudf::table compose_output_keys(cudf::table input_table) {
  std::vector<gdf_column*> output(input_table.num_columns() - 1);  
  std::transform(input_table.begin(), input_table.end() - 1, output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  return cudf::table {output};
}

rmm::device_vector<gdf_size_type> get_last_column (cudf::table current_table) {
  auto num_column = current_table.num_columns();
  gdf_column * sorted_column = current_table.get_column(num_column - 1);
  rmm::device_vector<gdf_size_type> returned_vector(current_table.num_rows());
  cudaMemcpy(returned_vector.data().get(), sorted_column->data, sorted_column->size * sizeof(gdf_size_type), cudaMemcpyDeviceToDevice); 
  return returned_vector;
}

std::pair<cudf::table, gdf_column> compute_sort_groupby_wo_agg(cudf::table const& input_keys, 
                            Options options,
                            rmm::device_vector<gdf_size_type> &d_sorted_indices,
                            cudaStream_t stream) {
  gdf_context context;
  auto ignore_null_keys = options.ignore_null_keys;
  if (not ignore_null_keys) { // SQL
    context.flag_groupby_include_nulls = true;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  } else { // PANDAS
    context.flag_groupby_include_nulls = false;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  }

  std::vector<int> groupby_col_indices;
  for (gdf_size_type i = 0; i < input_keys.num_columns(); i++)
    groupby_col_indices.push_back(i);

  cudf::table sorted_keys_table;
  gdf_column group_indices_col;
  
  auto nrows = input_keys.num_rows();
  rmm::device_vector<gdf_size_type> d_seq_indices_values(nrows);
  thrust::sequence(d_seq_indices_values.begin(), d_seq_indices_values.end(), 0, 1);

  gdf_column seq_indices_col{};
  CUDF_TRY(gdf_column_view(&seq_indices_col,
                           (void *)(d_seq_indices_values.data().get()), nullptr,
                           nrows, GDF_INT32));

  auto input_table = compose_inputs(input_keys, &seq_indices_col);
  std::tie(sorted_keys_table,
                        group_indices_col) = gdf_group_by_without_aggregations(input_table,
                                                                          groupby_col_indices.size(),
                                                                          groupby_col_indices.data(),
                                                                          &context);
  cudf::table output_keys = compose_output_keys(sorted_keys_table);
  d_sorted_indices = get_last_column(sorted_keys_table); 
  return std::make_pair(output_keys, group_indices_col);
}


struct median_result_type {
  template <typename SourceType>
  gdf_dtype operator()() {
    return cudf::gdf_dtype_of<target_type_t<SourceType, MIN>>(); // todo: MEDIAN: FIX THIS AFTER
  }
};

template <typename T>
struct compute_mediam_operator {
    gdf_size_type* map;
    gdf_size_type* map_group;
    T* values;
    
  __device__ inline T operator()(gdf_index_type i) {
    // gdf_size_type num_elements = get_num_elements_in_group(group_indices, i);
    gdf_size_type num_elements = 3; // FIX this 
    gdf_size_type index = num_elements * i; // FIX this index = index_container[i];
    gdf_size_type index_medium = index + num_elements / 2;
    return values[ map[ map_group[index_medium] ] ];
  }
};

struct dispath_mediam_forwarder {
  template <typename T>
  bool operator()(const gdf_column *input_column, gdf_column *output_column, rmm::device_vector<gdf_size_type> & map_indices_group, rmm::device_vector<gdf_size_type> & sorted_indices, cudaStream_t stream) const {
    compute_mediam_operator<T> op{sorted_indices.data().get(), map_indices_group.data().get(), (T*)input_column->data};
    thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator(gdf_size_type(0)), thrust::make_counting_iterator(gdf_size_type(map_indices_group.size())), (T *)output_column->data, op);
    return true;
  }
};

gdf_column* compute_median(const gdf_column& group_indices_col,  const gdf_column &input_column, rmm::device_vector<gdf_size_type> & map_indices_group, rmm::device_vector<gdf_size_type> & sorted_indices, cudaStream_t stream) {
  auto ngroups = group_indices_col.size;
  gdf_column* median = new gdf_column{};
  median->dtype = cudf::type_dispatcher(input_column.dtype, median_result_type{});
  const auto byte_width = (input_column.dtype == GDF_STRING)
                          ? sizeof(std::pair<const char *, size_t>)
                          : cudf::size_of(input_column.dtype);
  median->size = ngroups;
  RMM_TRY(RMM_ALLOC(&median->data, byte_width * ngroups, stream));
  if (cudf::is_nullable(input_column)) {
    RMM_TRY(RMM_ALLOC(
        &median->valid,
        sizeof(gdf_size_type) * gdf_valid_allocation_size(ngroups), stream));
  }
  cudf::type_dispatcher(input_column.dtype, dispath_mediam_forwarder{}, &input_column, median, map_indices_group, sorted_indices, stream);
  return median;
}

template <bool nullable = true>
struct row_group_inequality_comparator 
{
  row_inequality_comparator<nullable> cmp;
  gdf_size_type *index_map;
  gdf_size_type* group_indices; 
  gdf_size_type group_indices_size; 

    __device__ inline bool operator()(gdf_index_type x, gdf_index_type y) const
  {
    gdf_index_type group_id_x = get_group_index(group_indices, group_indices_size, x);
    gdf_index_type group_id_y = get_group_index(group_indices, group_indices_size, y);
    if (group_id_x == group_id_y) {
      return cmp(index_map[x], index_map[y]);
    }
    return false;
  }
};

gdf_error gdf_order_by_groups(const gdf_column& group_indices_col,
                       table input_keys,  
                       gdf_column const* const* cols,
                       int8_t* asc_desc,
                       size_t num_inputs,
                       gdf_column* output_indices,
                       rmm::device_vector<gdf_size_type> &d_index_map,
                       gdf_context * context)                       
{
  GDF_REQUIRE(cols != nullptr && output_indices != nullptr, GDF_DATASET_EMPTY);
  GDF_REQUIRE(cols[0]->size == output_indices->size, GDF_COLUMN_SIZE_MISMATCH);
  /* NOTE: providing support for indexes to be multiple different types explodes compilation time, such that it become infeasible */
  GDF_REQUIRE(output_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    
  bool nulls_are_smallest = false;
  if (context->flag_null_sort_behavior == GDF_NULL_AS_SMALLEST) {
  /* When sorting NULLS will be treated as the smallest number */
    nulls_are_smallest = true;
  } 
  
  cudaStream_t stream = 0;
  gdf_index_type* d_indx = static_cast<gdf_index_type*>(output_indices->data);
  gdf_size_type nrows = cols[0]->size;

  thrust::sequence(rmm::exec_policy(stream)->on(stream), d_indx, d_indx+nrows, 0);
  auto table = device_table::create(num_inputs, cols, stream);
  bool nullable = table.get()->has_nulls();
 
  // if (nullable){
  //   auto ineq_op = row_inequality_comparator<true>(*table, nulls_are_smallest, asc_desc); 
  //   thrust::sort(rmm::exec_policy(stream)->on(stream),
  //                 d_indx, d_indx+nrows,
  //                 ineq_op);				        
  // } else {
    // auto d_key_table = device_table::create(input_keys.num_columns(), input_keys.begin(), stream);
  // }

  auto ineq_op = row_inequality_comparator<false>(*table, nulls_are_smallest, asc_desc); 
  row_group_inequality_comparator<false> cmp {ineq_op, d_index_map.data().get(), (gdf_size_type*)group_indices_col.data, group_indices_col.size};
  thrust::sort(rmm::exec_policy(stream)->on(stream),
                d_indx, d_indx+nrows,
                cmp);				        
  
  return GDF_SUCCESS;
}


table compute_median_requests(
    std::vector<AggRequestType> const& original_requests,
    table input_keys,  
    const gdf_column& group_indices_col,
    rmm::device_vector<gdf_size_type> d_sorted_indices,
    cudaStream_t stream) { 
  std::vector<gdf_column*> final_value_columns(original_requests.size());

  // compute median operation 
  for (gdf_size_type i = 0; i < (gdf_size_type)original_requests.size(); i++)
  {
    const std::pair<gdf_column*, operators> element = original_requests[i];
    gdf_column * value_col = element.first;
    operators op = element.second;
    assert(op == MEDIAN);

    auto nrows = input_keys.num_rows(); 
    rmm::device_vector<gdf_size_type> map_indices_group(thrust::make_counting_iterator(int(0)),
                                      thrust::make_counting_iterator(int(nrows)));

    gdf_column sorted_indices_col{};
    CUDF_TRY(gdf_column_view(&sorted_indices_col,
                            (void*)(map_indices_group.data().get()), nullptr, nrows,
                            GDF_INT32));
    gdf_context ctxt{};
    gdf_order_by_groups(group_indices_col, input_keys, &value_col, nullptr, 1, &sorted_indices_col, d_sorted_indices, &ctxt);
    // auto ngroups = group_indices_col.size; 

    final_value_columns[i] = compute_median(group_indices_col, *value_col, map_indices_group, d_sorted_indices, stream);
  }
  return cudf::table{final_value_columns};
}
template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops, Options options,
                          cudaStream_t stream) {
  cudf::table sorted_keys_table;
  gdf_column group_indices_col;

  rmm::device_vector<gdf_size_type> d_sorted_indices;
  std::tie(sorted_keys_table,
                          group_indices_col) = compute_sort_groupby_wo_agg(input_keys, options, d_sorted_indices, stream);

  if (sorted_keys_table.num_rows() == 0) {
    return std::make_pair(
        cudf::empty_like(input_keys),
        cudf::table(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values)));
  }

  std::vector<AggRequestType> original_requests(input_values.num_columns());
  std::transform(input_values.begin(), input_values.end(), input_ops.begin(),
                 original_requests.begin(),
                 [](gdf_column const* col, operators op) {
                   return std::make_pair(const_cast<gdf_column*>(col), op);
                 });

  std::vector<SimpleAggRequestCounter> simple_requests =
      compound_to_simple(original_requests);

  std::vector<gdf_column*> simple_values_columns;
  std::vector<operators> simple_operators;
  for (auto const& p : simple_requests) {
    const AggRequestType& agg_req_type = p.first;
    simple_values_columns.push_back(
        const_cast<gdf_column*>(agg_req_type.first));
    simple_operators.push_back(agg_req_type.second);
  }
  // process simple columns
  if (simple_values_columns.size() > 0) {
    cudf::table simple_values_table{simple_values_columns};

    cudf::table simple_output_values{
        group_indices_col.size, target_dtypes(column_dtypes(simple_values_table), simple_operators),
        column_dtype_infos(simple_values_table), values_have_nulls, false, stream};

    initialize_with_identity(simple_output_values, simple_operators, stream);

    auto d_input_keys = device_table::create(sorted_keys_table);
    auto d_input_values = device_table::create(simple_values_table);
    auto d_output_values = device_table::create(simple_output_values, stream);
    rmm::device_vector<operators> d_ops(simple_operators);
  
    auto row_bitmask = cudf::row_bitmask(sorted_keys_table, stream);

    cudf::util::cuda::grid_config_1d grid_params{sorted_keys_table.num_rows(), 256};

    cudf::groupby::sort::aggregate_all_rows<keys_have_nulls, values_have_nulls><<<
        grid_params.num_blocks, grid_params.num_threads_per_block, 0, stream>>>(
        *d_input_keys, *d_input_values, *d_output_values, d_sorted_indices.data().get(), 
        (gdf_index_type *)group_indices_col.data, group_indices_col.size,
        d_ops.data().get(), row_bitmask.data().get());

    cudf::table destination_table(group_indices_col.size,
                                  cudf::column_dtypes(sorted_keys_table),
                                  cudf::column_dtype_infos(sorted_keys_table),
                                  keys_have_nulls);
    
    cudf::gather(&sorted_keys_table, (gdf_index_type *)group_indices_col.data,
                &destination_table); 

    cudf::table final_output_values = compute_original_requests(
        original_requests, simple_requests, simple_output_values, stream);

    // FIX: destroy temporal tables, and temporal gdf_columns! 
    return std::make_pair(destination_table, final_output_values);
  } 

  // process  median like  aggregation
  cudf::table destination_table(group_indices_col.size,
                                  cudf::column_dtypes(sorted_keys_table),
                                  cudf::column_dtype_infos(sorted_keys_table),
                                  keys_have_nulls);

  cudf::gather(&sorted_keys_table, (gdf_index_type *)group_indices_col.data,
                &destination_table); 

  cudf::table final_output_values = compute_median_requests(
        original_requests,  input_keys, group_indices_col, d_sorted_indices, stream);

  return std::make_pair(destination_table, final_output_values);
}

/**---------------------------------------------------------------------------*
 * @brief Returns appropriate callable instantiation of `compute_sort_groupby`
 * based on presence of null values in keys and values.
 *
 * @param keys The groupby key columns
 * @param values The groupby value columns
 * @return Instantiated callable of compute_sort_groupby
 *---------------------------------------------------------------------------**/
auto groupby_null_specialization(table const& keys, table const& values) {
  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<true, true>;
    } else {
      return compute_sort_groupby<true, false>;
    }
  } else {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<false, true>;
    } else {
      return compute_sort_groupby<false, false>;
    }
  }
}
} // anonymous namespace

namespace detail {

std::pair<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operators> const &ops,
                                            Options options,
                                            cudaStream_t stream) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");

  verify_operators(values, ops);

  // Empty inputs
  if (keys.num_rows() == 0) {
    return std::make_pair(
        cudf::empty_like(keys),
        cudf::table(0, target_dtypes(column_dtypes(values), ops), column_dtype_infos(values)));
  }

 auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  cudf::table output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, ops, options, stream);

  update_nvcategories(keys, output_keys, values, output_values);
  return std::make_pair(output_keys, output_values);
}

} // namespace detail

std::pair<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operators> const &ops,
                                            Options options) {
  return detail::groupby(keys, values, ops, options);
}

} // END: namespace sort
} // END: namespace groupby
} // END: namespace cudf
