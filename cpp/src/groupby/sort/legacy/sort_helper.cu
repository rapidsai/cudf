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

#include "sort_helper.hpp"

#include <copying/legacy/scatter.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/legacy/column_utils.hpp>
#include <utilities/legacy/cuda_utils.hpp>

#include <cudf/legacy/copying.hpp>

#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <tuple>
#include <numeric>

namespace {

/**
 * @brief Compares two `table` rows for equality as if the table were
 * ordered according to a specified permutation map.
 *
 */
template <bool nullable = true>
struct permuted_row_equality_comparator {
  row_equality_comparator<nullable> _comparator;
  cudf::size_type const *_map;

  /**
   * @brief Construct a permuted_row_equality_comparator.
   *
   * @param t The `table` whose rows will be compared
   * @param map The permutation map that specifies the effective ordering of
   *`t`. Must be the same size as `t.num_rows()`
   */
  permuted_row_equality_comparator(device_table const &t,
                                   cudf::size_type const *map)
      : _comparator(t, t, true), _map{map} {}

  /**
   * @brief Returns true if the two rows at the specified indices in the permuted
   * order are equivalent.
   *
   * For example, comparing rows `i` and `j` is
   * equivalent to comparing rows `map[i]` and `map[j]` in the original table.
   *
   * @param lhs The index of the first row
   * @param rhs The index of the second row
   * @returns if the two specified rows in the permuted order are equivalent
   */
  CUDA_DEVICE_CALLABLE
  bool operator()(cudf::size_type lhs, cudf::size_type rhs) {
    return _comparator(_map[lhs], _map[rhs]);
  }
};

} // namespace anonymous


namespace cudf {
namespace groupby {
namespace sort {
namespace detail { 

cudf::size_type helper::num_keys() {
  if (_num_keys > -1)
    return _num_keys;

  if (not _include_nulls and has_nulls(_keys)) {
    // The number of rows w/o null values `n` is indicated by number of valid bits
    // in the row bitmask. When `include_nulls == false`, then only rows `[0, n)` 
    // in the sorted order are considered for grouping. 
    CUDF_TRY(gdf_count_nonzero_mask(
      reinterpret_cast<cudf::valid_type*>(keys_row_bitmask().data().get()),
      _keys.num_rows(),
      &_num_keys));
  } else {
    _num_keys = _keys.num_rows();
  }

  return _num_keys; 
}

gdf_column const& helper::key_sort_order() {
  if (_key_sorted_order)
    return *_key_sorted_order;

  _key_sorted_order = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<cudf::size_type>(),
                      _keys.num_rows(),
                      false,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  if (_keys_pre_sorted) {
    auto d_key_sorted_order = static_cast<cudf::size_type*>(_key_sorted_order->data);

    thrust::sequence(rmm::exec_policy(_stream)->on(_stream), 
                     d_key_sorted_order,
                     d_key_sorted_order + _key_sorted_order->size, 0);

    return *_key_sorted_order;
  }
  
  gdf_context context{};
  context.flag_groupby_include_nulls = _include_nulls;
  context.flag_null_sort_behavior = (_null_sort_behavior == null_order::AFTER)
                                  ? GDF_NULL_AS_LARGEST
                                  : GDF_NULL_AS_SMALLEST;

  if (_include_nulls ||
      !cudf::has_nulls(_keys)) {  // SQL style
    CUDF_TRY(gdf_order_by(_keys.begin(), nullptr,
                          _keys.num_columns(), _key_sorted_order.get(),
                          &context));
  } else {  // Pandas style
   // Temporarily replace the first column's bitmask with one that indicates the 
   // presence of a null value within a row.  This allows moving all rows that contain
   // a null value to the end of the sorted order. 
    gdf_column null_row_representative = *(_keys.get_column(0));
    null_row_representative.valid =
        reinterpret_cast<cudf::valid_type*>(keys_row_bitmask().data().get());

    cudf::table keys{_keys};
    std::vector<gdf_column*> modified_keys(keys.begin(), keys.end());
    modified_keys[0] = &null_row_representative;
    cudf::table modified_keys_table(modified_keys.data(),
                                    modified_keys.size());

    CUDF_TRY(gdf_order_by(modified_keys_table.begin(), nullptr,
                          modified_keys_table.num_columns(),
                          _key_sorted_order.get(), &context));

    // All rows with one or more null values are at the end of the resulting sorted order.
  }

  return *_key_sorted_order;
}

rmm::device_vector<cudf::size_type> const& helper::group_offsets() {
  if (_group_offsets)
    return *_group_offsets;

  _group_offsets = std::make_unique<index_vector>(num_keys());

  auto device_input_table = device_table::create(_keys, _stream);
  auto sorted_order = static_cast<cudf::size_type*>(key_sort_order().data);
  decltype(_group_offsets->begin()) result_end;
  auto exec = rmm::exec_policy(_stream)->on(_stream);

  if (has_nulls(_keys)) {
    result_end = thrust::unique_copy(exec,
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(num_keys()),
      _group_offsets->begin(),
      permuted_row_equality_comparator<true>(*device_input_table, sorted_order));
  } else {
    result_end = thrust::unique_copy(exec, 
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(num_keys()),
      _group_offsets->begin(),
      permuted_row_equality_comparator<false>(*device_input_table, sorted_order));
  }

  cudf::size_type num_groups = thrust::distance(_group_offsets->begin(), result_end);
  _group_offsets->resize(num_groups);

  return *_group_offsets;
}

rmm::device_vector<cudf::size_type> const& helper::group_labels() {
  if (_group_labels)
    return *_group_labels;

  // Get group labels for future use in segmented sorting
  _group_labels = std::make_unique<index_vector>(num_keys());

  auto& group_labels = *_group_labels;
  auto exec = rmm::exec_policy(_stream)->on(_stream);
  thrust::scatter(exec,
    thrust::make_constant_iterator(1, decltype(num_groups())(1)), 
    thrust::make_constant_iterator(1, num_groups()), 
    group_offsets().begin() + 1, 
    group_labels.begin());
 
  thrust::inclusive_scan(exec,
                        group_labels.begin(),
                        group_labels.end(),
                        group_labels.begin());

  return group_labels;
}

gdf_column const& helper::unsorted_keys_labels() {
  if (_unsorted_keys_labels)
    return *_unsorted_keys_labels;

  _unsorted_keys_labels = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<cudf::size_type>(),
                      key_sort_order().size,
                      true,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  CUDA_TRY(cudaMemsetAsync(_unsorted_keys_labels->valid, 0,
                           gdf_num_bitmask_elements(_unsorted_keys_labels->size), 
                           _stream));
  
  gdf_column group_labels_col{};
  gdf_column_view(&group_labels_col, 
                  const_cast<cudf::size_type*>(group_labels().data().get()), 
                  nullptr,
                  group_labels().size(), 
                  gdf_dtype_of<cudf::size_type>());
  cudf::table t_sorted_labels{&group_labels_col};
  cudf::table t_unsorted_keys_labels{_unsorted_keys_labels.get()};
  cudf::detail::scatter(&t_sorted_labels,
                        static_cast<cudf::size_type*>(key_sort_order().data),
                        &t_unsorted_keys_labels);
  return *_unsorted_keys_labels;
}

rmm::device_vector<bit_mask::bit_mask_t>&
helper::keys_row_bitmask() {
  if (_keys_row_bitmask)
    return *_keys_row_bitmask;

  _keys_row_bitmask = 
    bitmask_vec_pointer( new bitmask_vector(row_bitmask(_keys, _stream)));

  return *_keys_row_bitmask;
}

std::pair<gdf_column, rmm::device_vector<cudf::size_type> >
helper::sort_values(gdf_column const& values) {
  CUDF_EXPECTS(values.size == _keys.num_rows(),
    "Size mismatch between keys and values.");
  auto values_sort_order = gdf_col_pointer(
    new gdf_column(
      allocate_column(gdf_dtype_of<cudf::size_type>(),
                      _keys.num_rows(),
                      false,
                      gdf_dtype_extra_info{},
                      _stream)),
    [](gdf_column* col) { gdf_column_free(col); });

  // Need to const_cast because there cannot be a table constructor that can 
  // take const initializer list. Making separate constructors for const objects
  // is not supported in C++14 https://stackoverflow.com/a/49151864/3325146
  auto unsorted_values = const_cast<gdf_column*> (&values);
  auto unsorted_label_col = const_cast<gdf_column*> (&unsorted_keys_labels());
  auto unsorted_table = cudf::table{unsorted_label_col, unsorted_values};

  gdf_context context{};
  context.flag_groupby_include_nulls = _include_nulls;
  gdf_order_by(unsorted_table.begin(),
              nullptr,
              unsorted_table.num_columns(), // always 2
              values_sort_order.get(),
              &context);

  cudf::table unsorted_values_table{unsorted_values};
  auto sorted_values = allocate_like(values, num_keys(), RETAIN, _stream);
  cudf::table sorted_values_table{&sorted_values};
  cudf::gather(&unsorted_values_table,
              static_cast<cudf::size_type*>(values_sort_order->data),
              &sorted_values_table);

  // Get number of valid values in each group
  rmm::device_vector<cudf::size_type> val_group_sizes(num_groups());
  auto col_valid = reinterpret_cast<bit_mask::bit_mask_t*>(sorted_values.valid);
  
  auto bitmask_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), 
    [col_valid] __device__ (cudf::size_type i) -> int { 
      return (col_valid) ? bit_mask::is_valid(col_valid, i) : true;
    });

  thrust::reduce_by_key(rmm::exec_policy(_stream)->on(_stream),
                        group_labels().begin(),
                        group_labels().end(),
                        bitmask_iterator,
                        thrust::make_discard_iterator(),
                        val_group_sizes.begin());

  return std::make_pair(std::move(sorted_values), std::move(val_group_sizes));
}

cudf::table helper::unique_keys() {
  cudf::table unique_keys = allocate_like(_keys, 
                                          (cudf::size_type)num_groups(),
                                          RETAIN,
                                          _stream);
  auto idx_data = static_cast<cudf::size_type*>(key_sort_order().data);
  auto transformed_group_ids = index_vector(num_groups());

  auto exec = rmm::exec_policy(_stream)->on(_stream);

  thrust::transform(exec, group_offsets().begin(), group_offsets().end(),
                    transformed_group_ids.begin(),
    [=] __device__ (cudf::size_type i) { return idx_data[i]; } );
  
  cudf::gather(&_keys,
              transformed_group_ids.data().get(),
              &unique_keys);
  return unique_keys;
}


}  // namespace detail
}  // namespace sort
}  // namespace groupby
}  // namespace cudf
