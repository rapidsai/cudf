/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace {
/**
 * @brief Compares two `table` rows for equality as if the table were
 * ordered according to a specified permutation map.
 */
struct permuted_row_equality_comparator {
  cudf::row_equality_comparator<cudf::nullate::DYNAMIC> _comparator;
  cudf::size_type const* _map;

  /**
   * @brief Construct a permuted_row_equality_comparator.
   *
   * @param t The `table` whose rows will be compared
   * @param map The permutation map that specifies the effective ordering of
   * `t`. Must be the same size as `t.num_rows()`
   */
  permuted_row_equality_comparator(cudf::table_device_view const& t,
                                   cudf::size_type const* map,
                                   bool nullable = true)
    : _comparator(cudf::nullate::DYNAMIC{nullable}, t, t, cudf::null_equality::EQUAL), _map{map}
  {
  }

  /**
   * @brief Returns true if the two rows at the specified indices in the permuted
   * order are equivalent.
   *
   * For example, comparing rows `i` and `j` is
   * equivalent to comparing rows `map[i]` and `map[j]` in the original table.
   *
   * @param lhs The index of the first row
   * @param rhs The index of the second row
   * @returns true if the two specified rows in the permuted order are equivalent
   */
  __device__ inline bool operator()(cudf::size_type lhs, cudf::size_type rhs)
  {
    return _comparator(_map[lhs], _map[rhs]);
  }
};

}  // namespace

namespace cudf {
namespace groupby {
namespace detail {
namespace sort {

sort_groupby_helper::sort_groupby_helper(table_view const& keys,
                                         null_policy include_null_keys,
                                         sorted keys_pre_sorted)
  : _unflattened_keys(keys),
    _num_keys(-1),
    _keys_pre_sorted(keys_pre_sorted),
    _include_null_keys(include_null_keys)
{
  using namespace cudf::structs::detail;

  _flattened                 = flatten_nested_columns(keys, {}, {}, column_nullability::FORCE);
  _keys                      = _flattened;
  auto is_supported_key_type = [](auto col) { return cudf::is_equality_comparable(col.type()); };
  CUDF_EXPECTS(std::all_of(_keys.begin(), _keys.end(), is_supported_key_type),
               "Unsupported groupby key type does not support equality comparison");

  // Cannot depend on caller's sorting if the column contains nulls,
  // and null values are to be excluded.
  // Re-sort the data, to filter out nulls more easily.
  if (keys_pre_sorted == sorted::YES and include_null_keys == null_policy::EXCLUDE and
      has_nulls(keys)) {
    _keys_pre_sorted = sorted::NO;
  }
};

size_type sort_groupby_helper::num_keys(rmm::cuda_stream_view stream)
{
  if (_num_keys > -1) return _num_keys;

  if (_include_null_keys == null_policy::EXCLUDE and has_nulls(_keys)) {
    // The number of rows w/o null values `n` is indicated by number of valid bits
    // in the row bitmask. When `_include_null_keys == NO`, then only rows `[0, n)`
    // in the sorted keys are considered for grouping.
    _num_keys = keys_bitmask_column(stream).size() - keys_bitmask_column(stream).null_count();
  } else {
    _num_keys = _keys.num_rows();
  }

  return _num_keys;
}

column_view sort_groupby_helper::key_sort_order(rmm::cuda_stream_view stream)
{
  auto sliced_key_sorted_order = [stream, this]() {
    return cudf::detail::slice(this->_key_sorted_order->view(), 0, this->num_keys(stream));
  };

  if (_key_sorted_order) { return sliced_key_sorted_order(); }

  // TODO (dm): optimization. When keys are pre sorted but ignore nulls is true,
  //            we still want all rows with nulls in the end. Sort is costly, so
  //            do a copy_if(counting, sorted_order, {bitmask.is_valid(i)})
  if (_keys_pre_sorted == sorted::YES) {
    _key_sorted_order = make_numeric_column(
      data_type(type_to_id<size_type>()), _keys.num_rows(), mask_state::UNALLOCATED, stream);

    auto d_key_sorted_order = _key_sorted_order->mutable_view().data<size_type>();

    thrust::sequence(rmm::exec_policy(stream),
                     d_key_sorted_order,
                     d_key_sorted_order + _key_sorted_order->size(),
                     0);

    return sliced_key_sorted_order();
  }

  if (_include_null_keys == null_policy::INCLUDE || !cudf::has_nulls(_keys)) {  // SQL style
    _key_sorted_order = cudf::detail::stable_sorted_order(
      _keys,
      {},
      std::vector<null_order>(_keys.num_columns(), null_order::AFTER),
      stream,
      rmm::mr::get_current_device_resource());
  } else {  // Pandas style
    // Temporarily prepend the keys table with a column that indicates the
    // presence of a null value within a row. This allows moving all rows that
    // contain a null value to the end of the sorted order.

    auto augmented_keys = table_view({table_view({keys_bitmask_column(stream)}), _keys});

    _key_sorted_order = cudf::detail::stable_sorted_order(
      augmented_keys,
      {},
      std::vector<null_order>(_keys.num_columns() + 1, null_order::AFTER),
      stream,
      rmm::mr::get_current_device_resource());

    // All rows with one or more null values are at the end of the resulting sorted order.
  }

  return sliced_key_sorted_order();
}

sort_groupby_helper::index_vector const& sort_groupby_helper::group_offsets(
  rmm::cuda_stream_view stream)
{
  if (_group_offsets) return *_group_offsets;

  _group_offsets = std::make_unique<index_vector>(num_keys(stream) + 1, stream);

  auto device_input_table = table_device_view::create(_keys, stream);
  auto sorted_order       = key_sort_order(stream).data<size_type>();
  decltype(_group_offsets->begin()) result_end;

  result_end = thrust::unique_copy(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_keys(stream)),
    _group_offsets->begin(),
    permuted_row_equality_comparator(*device_input_table, sorted_order, has_nulls(_keys)));

  size_type num_groups = thrust::distance(_group_offsets->begin(), result_end);
  _group_offsets->set_element(num_groups, num_keys(stream), stream);
  _group_offsets->resize(num_groups + 1, stream);

  return *_group_offsets;
}

sort_groupby_helper::index_vector const& sort_groupby_helper::group_labels(
  rmm::cuda_stream_view stream)
{
  if (_group_labels) return *_group_labels;

  // Get group labels for future use in segmented sorting
  _group_labels = std::make_unique<index_vector>(num_keys(stream), stream);

  auto& group_labels = *_group_labels;
  if (num_keys(stream) == 0) return group_labels;

  cudf::detail::label_segments(group_offsets(stream).begin(),
                               group_offsets(stream).end(),
                               group_labels.begin(),
                               group_labels.end(),
                               stream);
  return group_labels;
}

column_view sort_groupby_helper::unsorted_keys_labels(rmm::cuda_stream_view stream)
{
  if (_unsorted_keys_labels) return _unsorted_keys_labels->view();

  column_ptr temp_labels = make_numeric_column(
    data_type(type_to_id<size_type>()), _keys.num_rows(), mask_state::ALL_NULL, stream);

  auto group_labels_view = cudf::column_view(
    data_type(type_to_id<size_type>()), group_labels(stream).size(), group_labels(stream).data());

  auto scatter_map = key_sort_order(stream);

  std::unique_ptr<table> t_unsorted_keys_labels =
    cudf::detail::scatter(table_view({group_labels_view}),
                          scatter_map,
                          table_view({temp_labels->view()}),
                          stream,
                          rmm::mr::get_current_device_resource());

  _unsorted_keys_labels = std::move(t_unsorted_keys_labels->release()[0]);

  return _unsorted_keys_labels->view();
}

column_view sort_groupby_helper::keys_bitmask_column(rmm::cuda_stream_view stream)
{
  if (_keys_bitmask_column) return _keys_bitmask_column->view();

  auto [row_bitmask, null_count] = cudf::detail::bitmask_and(_keys, stream);

  _keys_bitmask_column = make_numeric_column(
    data_type(type_id::INT8), _keys.num_rows(), std::move(row_bitmask), null_count, stream);

  auto keys_bitmask_view = _keys_bitmask_column->mutable_view();
  using T                = id_to_type<type_id::INT8>;
  thrust::fill(
    rmm::exec_policy(stream), keys_bitmask_view.begin<T>(), keys_bitmask_view.end<T>(), 0);

  return _keys_bitmask_column->view();
}

sort_groupby_helper::column_ptr sort_groupby_helper::sorted_values(
  column_view const& values, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  column_ptr values_sort_order =
    cudf::detail::stable_sorted_order(table_view({unsorted_keys_labels(stream), values}),
                                      {},
                                      std::vector<null_order>(2, null_order::AFTER),
                                      stream,
                                      mr);

  // Zero-copy slice this sort order so that its new size is num_keys()
  column_view gather_map = cudf::detail::slice(values_sort_order->view(), 0, num_keys(stream));

  auto sorted_values_table = cudf::detail::gather(table_view({values}),
                                                  gather_map,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                  stream,
                                                  mr);

  return std::move(sorted_values_table->release()[0]);
}

sort_groupby_helper::column_ptr sort_groupby_helper::grouped_values(
  column_view const& values, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  auto gather_map = key_sort_order(stream);

  auto grouped_values_table = cudf::detail::gather(table_view({values}),
                                                   gather_map,
                                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                                   cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                   stream,
                                                   mr);

  return std::move(grouped_values_table->release()[0]);
}

std::unique_ptr<table> sort_groupby_helper::unique_keys(rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource* mr)
{
  auto idx_data = key_sort_order(stream).data<size_type>();

  auto gather_map_it = thrust::make_transform_iterator(
    group_offsets(stream).begin(), [idx_data] __device__(size_type i) { return idx_data[i]; });

  return cudf::detail::gather(_unflattened_keys,
                              gather_map_it,
                              gather_map_it + num_groups(stream),
                              out_of_bounds_policy::DONT_CHECK,
                              stream,
                              mr);
}

std::unique_ptr<table> sort_groupby_helper::sorted_keys(rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource* mr)
{
  return cudf::detail::gather(_unflattened_keys,
                              key_sort_order(stream),
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
