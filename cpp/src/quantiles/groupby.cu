/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "groupby.hpp"

#include <copying/scatter.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include <cudf/copying.hpp>

#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <algorithm>
#include <tuple>
#include <numeric>

namespace {

template <bool nullable = true>
struct transform_row_eq_comparator {
  row_equality_comparator<nullable> cmp;
  gdf_size_type const* sorted_order;

  CUDA_DEVICE_CALLABLE
  bool operator() (gdf_size_type lhs, gdf_size_type rhs) {
    return cmp(sorted_order[lhs], sorted_order[rhs]);
  }
};

} // namespace anonymous


namespace cudf {

namespace detail {

gdf_column const& groupby::key_sort_order() {
  if (_key_sorted_order)
    return *_key_sorted_order;

  _key_sorted_order = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<gdf_index_type>(),
                      _keys.num_rows(),
                      false,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  if (_include_nulls ||
      !cudf::has_nulls(_keys)) {  // SQL style
    gdf_context context{};
    context.flag_groupby_include_nulls = true;
    CUDF_TRY(gdf_order_by(_keys.begin(), nullptr,
                          _keys.num_columns(), _key_sorted_order.get(),
                          &context));
  } else {  // Pandas style
    auto keys_row_bitmask = row_bitmask(_keys, _stream);

    gdf_column null_row_representative = *(_keys.get_column(0));
    null_row_representative.valid =
        reinterpret_cast<gdf_valid_type*>(keys_row_bitmask.data().get());

    cudf::table keys{_keys};
    std::vector<gdf_column*> modified_keys(keys.begin(), keys.end());
    modified_keys[0] = &null_row_representative;
    cudf::table modified_keys_table(modified_keys.data(),
                                    modified_keys.size());

    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

    CUDF_TRY(gdf_order_by(modified_keys_table.begin(), nullptr,
                          modified_keys_table.num_columns(),
                          _key_sorted_order.get(), &temp_ctx));

    CUDF_TRY(gdf_count_nonzero_mask(
        reinterpret_cast<gdf_valid_type*>(keys_row_bitmask.data().get()),
        _keys.num_rows(),
        &_num_keys));
  }

  return *_key_sorted_order;
}

rmm::device_vector<gdf_size_type> const& groupby::group_offsets() {
  if (_group_offsets)
    return *_group_offsets;

  _group_offsets = std::make_unique<index_vector>(num_keys());

  auto device_input_table = device_table::create(_keys, _stream);
  auto sorted_order = static_cast<gdf_size_type*>(key_sort_order().data);
  decltype(_group_offsets->begin()) result_end;
  auto exec = rmm::exec_policy(_stream)->on(_stream);

  if (has_nulls(_keys)) {
    auto comp = row_equality_comparator<true>(*device_input_table, true);
    result_end = thrust::unique_copy(exec,
      thrust::make_counting_iterator<gdf_size_type>(0),
      thrust::make_counting_iterator<gdf_size_type>(num_keys()),
      _group_offsets->begin(), transform_row_eq_comparator<true>{comp, sorted_order});
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table, true);
    result_end = thrust::unique_copy(exec, 
      thrust::make_counting_iterator<gdf_size_type>(0),
      thrust::make_counting_iterator<gdf_size_type>(num_keys()),
      _group_offsets->begin(), transform_row_eq_comparator<false>{comp, sorted_order});
  }

  gdf_size_type num_groups = thrust::distance(_group_offsets->begin(), result_end);
  _group_offsets->resize(num_groups);

  return *_group_offsets;
}

rmm::device_vector<gdf_size_type> const& groupby::group_labels() {
  if (_group_labels)
    return *_group_labels;

  // Get group labels for future use in segmented sorting
  _group_labels = std::make_unique<index_vector>(num_keys());

  auto& group_labels = *_group_labels;
  auto exec = rmm::exec_policy(_stream)->on(_stream);
  auto group_labels_ptr = group_labels.data().get();
  auto group_offsets_ptr = group_offsets().data().get();
  thrust::for_each_n(exec,
    thrust::make_counting_iterator(1),
    num_groups() - 1,
    [group_labels_ptr, group_offsets_ptr] __device__ (gdf_size_type i) {
      group_labels_ptr[group_offsets_ptr[i]] = 1;
    });
  thrust::inclusive_scan(exec,
                        group_labels.begin(),
                        group_labels.end(),
                        group_labels.begin());

  return group_labels;
}

gdf_column const& groupby::unsorted_labels() {
  if (_unsorted_labels)
    return *_unsorted_labels;

  _unsorted_labels = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<gdf_size_type>(),
                      key_sort_order().size,
                      true,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  CUDA_TRY(cudaMemsetAsync(_unsorted_labels->valid, 0,
                           gdf_num_bitmask_elements(_unsorted_labels->size), 
                           _stream));
  
  gdf_column group_labels_col{};
  gdf_column_view(&group_labels_col, 
                  const_cast<gdf_size_type*>(group_labels().data().get()), 
                  nullptr,
                  group_labels().size(), 
                  gdf_dtype_of<gdf_size_type>());
  cudf::table t_sorted_labels{&group_labels_col};
  cudf::table t_unsorted_labels{_unsorted_labels.get()};
  cudf::detail::scatter(&t_sorted_labels,
                        static_cast<gdf_size_type*>(key_sort_order().data),
                        &t_unsorted_labels);
  return *_unsorted_labels;
}


std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
groupby::sort_values(gdf_column const& values) {
  auto values_sort_order = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<gdf_index_type>(),
                      _keys.num_rows(),
                      false,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  // We need a table constructor that can take const initializer list
  auto unsorted_values = const_cast<gdf_column*> (&values);
  auto unsorted_label_col = const_cast<gdf_column*> (&unsorted_labels());
  auto unsorted_table = cudf::table{unsorted_label_col, unsorted_values};

  gdf_context context{};
  context.flag_groupby_include_nulls = _include_nulls;
  gdf_order_by(unsorted_table.begin(),
              nullptr,
              unsorted_table.num_columns(), // always 2
              values_sort_order.get(),
              &context);

  cudf::table unsorted_values_table{unsorted_values};
  auto sorted_values = allocate_like(values, num_keys(), true, _stream);
  cudf::table sorted_values_table{&sorted_values};
  cudf::gather(&unsorted_values_table,
              static_cast<gdf_size_type*>(values_sort_order->data),
              &sorted_values_table);

  // Get number of valid values in each group
  rmm::device_vector<gdf_size_type> val_group_sizes(num_groups());
  auto col_valid = reinterpret_cast<bit_mask::bit_mask_t*>(sorted_values.valid);
  
  auto bitmask_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), 
    [col_valid] __device__ (gdf_size_type i) -> int { 
      return (col_valid) ? bit_mask::is_valid(col_valid, i) : true;
    });

  thrust::reduce_by_key(rmm::exec_policy(_stream)->on(_stream),
                        group_labels().begin(),
                        group_labels().end(),
                        bitmask_iterator,
                        thrust::make_discard_iterator(),
                        val_group_sizes.begin());

  return std::make_pair(sorted_values, val_group_sizes);
}

cudf::table groupby::unique_keys() {
  cudf::table unique_keys = allocate_like(_keys, 
                                          (gdf_size_type)num_groups(),
                                          true,
                                          _stream);
  auto idx_data = static_cast<gdf_size_type*>(key_sort_order().data);
  auto transformed_group_ids = index_vector(num_groups());

  auto exec = rmm::exec_policy(_stream)->on(_stream);

  thrust::transform(exec, group_offsets().begin(), group_offsets().end(),
                    transformed_group_ids.begin(),
    [=] __device__ (gdf_size_type i) { return idx_data[i]; } );
  
  cudf::gather(&_keys,
              transformed_group_ids.data().get(),
              &unique_keys);
  return unique_keys;
}


} // namespace detail
  
} // namespace cudf
