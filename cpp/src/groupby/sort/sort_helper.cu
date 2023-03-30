/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "common_utils.cuh"
#include "cudf/column/column_device_view.cuh"
#include "cudf/detail/utilities/vector_factories.hpp"
#include "cudf/null_mask.hpp"
#include "cudf_test/column_utilities.hpp"
#include "thrust/detail/copy.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/unique.h>

#include <cuda/functional>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace cudf {
namespace groupby {
namespace detail {
namespace sort {

sort_groupby_helper::sort_groupby_helper(table_view const& keys,
                                         null_policy include_null_keys,
                                         sorted keys_pre_sorted,
                                         std::vector<null_order> const& null_precedence)
  : _keys(keys),
    _num_keys(-1),
    _keys_pre_sorted(keys_pre_sorted),
    _include_null_keys(include_null_keys),
    _null_precedence(null_precedence)
{
  // Cannot depend on caller's sorting if the column contains nulls,
  // and null values are to be excluded.
  // Re-sort the data, to filter out nulls more easily.
  if (keys_pre_sorted == sorted::YES and include_null_keys == null_policy::EXCLUDE and
      has_nulls(keys)) {
    _keys_pre_sorted = sorted::NO;
  }
  if (_keys_pre_sorted == sorted::YES)
    is_using_hashing = false;
  else
    is_using_hashing = true;
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

void print_view(column_view view, rmm::cuda_stream_view stream)
{
  // cudf::test::print(view);
}

void sort_groupby_helper::hash_sorter(rmm::cuda_stream_view stream)
{
  using namespace cudf::detail;
  auto input = (_include_null_keys == null_policy::INCLUDE || !cudf::has_nulls(_keys))
                 ?  // SQL style
                 _keys
                 : table_view({table_view({keys_bitmask_column(stream)}), _keys});
  constexpr auto nulls_equal = null_equality::EQUAL;
  auto map                   = hash_map_type{compute_hash_table_size(input.num_rows()),
                           cuco::empty_key{COMPACTION_EMPTY_KEY_SENTINEL},
                           cuco::empty_value{COMPACTION_EMPTY_VALUE_SENTINEL},
                           hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls          = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher =
    cudf::detail::experimental::compaction_hash(row_hasher.device_hasher(has_nulls));

  auto const row_comp = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  // TODO: return value as std::numerical_limits::max() for nulls so that it goes to last while
  // sorting labels.
  auto const pair_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });
  auto const count_iter = thrust::make_counting_iterator(size_type{0});

  using nan_equal_comparator =
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
  nan_equal_comparator value_comp{};

  _unsorted_keys_labels = make_numeric_column(
    data_type(type_to_id<size_type>()), _keys.num_rows(), mask_state::UNALLOCATED, stream);
  auto unsorted_keys_labels_begin = _unsorted_keys_labels->mutable_view().data<size_type>();

  if (has_nested_columns) {
    auto const key_equal = row_comp.equal_to<true>(has_nulls, nulls_equal, value_comp);
    // should I use insert_if?
    // if (_include_null_keys == null_policy::EXCLUDE and has_nulls(_keys));
    map.insert(pair_iter, pair_iter + input.num_rows(), key_hasher, key_equal, stream.value());
    map.find(count_iter,
             count_iter + input.num_rows(),
             unsorted_keys_labels_begin,
             key_hasher,
             key_equal,
             stream.value());
  } else {
    auto const key_equal = row_comp.equal_to<false>(has_nulls, nulls_equal, value_comp);
    map.insert(pair_iter, pair_iter + input.num_rows(), key_hasher, key_equal, stream.value());
    map.find(count_iter,
             count_iter + input.num_rows(),
             unsorted_keys_labels_begin,
             key_hasher,
             key_equal,
             stream.value());
    // TODO check if insert_and_find device function is faster, if so, use it
  }
  // how to find null's label and exclude it?
  // copy bitmask to _unsorted_keys_labels, and use it to sort for _key_sorted_order, and also
  // group_labels.
  auto const any_nulls = _include_null_keys == null_policy::EXCLUDE and cudf::has_nulls(_keys);
  if (any_nulls)
    _unsorted_keys_labels->set_null_mask(
      cudf::detail::copy_bitmask(
        keys_bitmask_column(stream), stream, rmm::mr::get_current_device_resource()),
      keys_bitmask_column(stream).null_count());
  auto sort_key = _unsorted_keys_labels->view();
  // auto _unsorted_keys_labels_view = _unsorted_keys_labels->view();
  // auto sort_key = column_view{_unsorted_keys_labels_view.type(),
  //             _unsorted_keys_labels_view.size(),
  //             _unsorted_keys_labels_view.head(),
  //             any_nulls ? keys_bitmask_column(stream).null_mask() :
  //             _unsorted_keys_labels_view.null_mask(), any_nulls ?
  //             keys_bitmask_column(stream).null_count() : _unsorted_keys_labels_view.null_count(),
  //             _unsorted_keys_labels_view.offset()};
  // pushes nulls to last.
  _key_sorted_order = cudf::detail::stable_sorted_order(table_view{{sort_key}},
                                                        {},
                                                        {null_order::AFTER},
                                                        stream,
                                                        rmm::mr::get_current_device_resource());
  // print_view(_key_sorted_order->view(), stream); std::cout<<"^_kso\n";
  // print_view(_unsorted_keys_labels->view(), stream); std::cout<<"^_ukl\n";
}

column_view sort_groupby_helper::key_sort_order(rmm::cuda_stream_view stream)
{
  auto sliced_key_sorted_order = [stream, this]() {
    return cudf::detail::slice(this->_key_sorted_order->view(), 0, this->num_keys(stream), stream);
  };

  if (_key_sorted_order) { return sliced_key_sorted_order(); }

  if (_keys_pre_sorted == sorted::YES) {
    _key_sorted_order = cudf::detail::sequence(_keys.num_rows(),
                                               numeric_scalar<size_type>(0),
                                               numeric_scalar<size_type>(1),
                                               stream,
                                               rmm::mr::get_current_device_resource());
    return sliced_key_sorted_order();
  }

  // if (is_using_hashing)
  if (std::getenv("USE_HASHING"))
    hash_sorter(stream);
  else if (_include_null_keys == null_policy::INCLUDE || !cudf::has_nulls(_keys)) {  // SQL style
    auto const precedence = _null_precedence.empty()
                              ? std::vector(_keys.num_columns(), null_order::AFTER)
                              : _null_precedence;
    _key_sorted_order     = cudf::detail::stable_sorted_order(
      _keys, {}, precedence, stream, rmm::mr::get_current_device_resource());
  } else {  // Pandas style
    // Temporarily prepend the keys table with a column that indicates the
    // presence of a null value within a row. This allows moving all rows that
    // contain a null value to the end of the sorted order.

    auto const augmented_keys = table_view({table_view({keys_bitmask_column(stream)}), _keys});
    auto const precedence     = [&]() {
      auto precedence = _null_precedence.empty()
                              ? std::vector<null_order>(_keys.num_columns(), null_order::AFTER)
                              : _null_precedence;
      precedence.insert(precedence.begin(), null_order::AFTER);
      return precedence;
    }();

    _key_sorted_order = cudf::detail::stable_sorted_order(
      augmented_keys, {}, precedence, stream, rmm::mr::get_current_device_resource());

    // All rows with one or more null values are at the end of the resulting sorted order.
  }

  print_view(sliced_key_sorted_order(), stream);
  std::cout << "^kso\n";
  print_view(unsorted_keys_labels(stream), stream);
  std::cout << "^ukl\n";
  return sliced_key_sorted_order();
}

sort_groupby_helper::index_vector const& sort_groupby_helper::group_offsets(
  rmm::cuda_stream_view stream)
{
  if (_group_offsets) return *_group_offsets;

  auto const size = num_keys(stream);
  // Create a temporary variable and only set _group_offsets right before the return.
  // This way, a 2nd (parallel) call to this will not be given a partially created object.
  auto group_offsets = std::make_unique<index_vector>(size + 1, stream);

  auto const comparator = cudf::experimental::row::equality::self_comparator{_keys, stream};

  auto const sorted_order = key_sort_order(stream).data<size_type>();
  decltype(group_offsets->begin()) result_end;

  if (cudf::detail::has_nested_columns(_keys)) {
    auto const d_key_equal = comparator.equal_to<true>(
      cudf::nullate::DYNAMIC{cudf::has_nested_nulls(_keys)}, null_equality::EQUAL);
    // Using a temporary buffer for intermediate transform results from the iterator containing
    // the comparator speeds up compile-time significantly without much degradation in
    // runtime performance over using the comparator directly in thrust::unique_copy.
    auto result       = rmm::device_uvector<bool>(size, stream);
    auto const itr    = thrust::make_counting_iterator<size_type>(0);
    auto const row_eq = permuted_row_equality_comparator(d_key_equal, sorted_order);
    auto const ufn    = cudf::detail::unique_copy_fn<decltype(itr), decltype(row_eq)>{
      itr, duplicate_keep_option::KEEP_FIRST, row_eq, size - 1};
    thrust::transform(rmm::exec_policy(stream), itr, itr + size, result.begin(), ufn);
    result_end = thrust::copy_if(rmm::exec_policy(stream),
                                 itr,
                                 itr + size,
                                 result.begin(),
                                 group_offsets->begin(),
                                 thrust::identity<bool>{});
  } else {
    auto const d_key_equal = comparator.equal_to<false>(
      cudf::nullate::DYNAMIC{cudf::has_nested_nulls(_keys)}, null_equality::EQUAL);
    result_end = thrust::unique_copy(rmm::exec_policy(stream),
                                     thrust::counting_iterator<size_type>(0),
                                     thrust::counting_iterator<size_type>(size),
                                     group_offsets->begin(),
                                     permuted_row_equality_comparator(d_key_equal, sorted_order));
  }

  auto const num_groups = thrust::distance(group_offsets->begin(), result_end);
  group_offsets->set_element_async(num_groups, size, stream);
  group_offsets->resize(num_groups + 1, stream);

  _group_offsets = std::move(group_offsets);
  return *_group_offsets;
}

sort_groupby_helper::index_vector const& sort_groupby_helper::group_labels(
  rmm::cuda_stream_view stream)
{
  if (_group_labels) return *_group_labels;

  // Create a temporary variable and only set _group_labels right before the return.
  // This way, a 2nd (parallel) call to this will not be given a partially created object.
  auto group_labels = std::make_unique<index_vector>(num_keys(stream), stream);

  if (num_keys(stream)) {
    auto const& offsets = group_offsets(stream);
    cudf::detail::label_segments(
      offsets.begin(), offsets.end(), group_labels->begin(), group_labels->end(), stream);
  }

  _group_labels = std::move(group_labels);
  return *_group_labels;
}

column_view sort_groupby_helper::unsorted_keys_labels(rmm::cuda_stream_view stream)
{
  if (_unsorted_keys_labels) {
    // print_view(_unsorted_keys_labels->view(), stream); std::cout<<"^ukl\n";
    return _unsorted_keys_labels->view();
  }

  column_ptr temp_labels = make_numeric_column(
    data_type(type_to_id<size_type>()), _keys.num_rows(), mask_state::ALL_NULL, stream);

  auto group_labels_view = cudf::column_view(data_type(type_to_id<size_type>()),
                                             group_labels(stream).size(),
                                             group_labels(stream).data(),
                                             nullptr,
                                             0);

  auto scatter_map = key_sort_order(stream);

  std::unique_ptr<table> t_unsorted_keys_labels =
    cudf::detail::scatter(table_view({group_labels_view}),
                          scatter_map,
                          table_view({temp_labels->view()}),
                          stream,
                          rmm::mr::get_current_device_resource());

  _unsorted_keys_labels = std::move(t_unsorted_keys_labels->release()[0]);

  // print_view(_unsorted_keys_labels->view(), stream); std::cout<<"^ukl\n";
  return _unsorted_keys_labels->view();
}

column_view sort_groupby_helper::keys_bitmask_column(rmm::cuda_stream_view stream)
{
  if (_keys_bitmask_column) return _keys_bitmask_column->view();

  auto [row_bitmask, null_count] =
    cudf::detail::bitmask_and(_keys, stream, rmm::mr::get_current_device_resource());

  auto const zero = numeric_scalar<int8_t>(0);
  // Create a temporary variable and only set _keys_bitmask_column right before the return.
  // This way, a 2nd (parallel) call to this will not be given a partially created object.
  auto keys_bitmask_column = cudf::detail::sequence(
    _keys.num_rows(), zero, zero, stream, rmm::mr::get_current_device_resource());
  keys_bitmask_column->set_null_mask(std::move(row_bitmask), null_count);

  _keys_bitmask_column = std::move(keys_bitmask_column);
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
  column_view gather_map =
    cudf::detail::slice(values_sort_order->view(), 0, num_keys(stream), stream);

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

  auto gather_map_it =
    thrust::make_transform_iterator(group_offsets(stream).begin(),
                                    cuda::proclaim_return_type<size_type>(
                                      [idx_data] __device__(size_type i) { return idx_data[i]; }));

  return cudf::detail::gather(_keys,
                              gather_map_it,
                              gather_map_it + num_groups(stream),
                              out_of_bounds_policy::DONT_CHECK,
                              stream,
                              mr);
}

std::unique_ptr<table> sort_groupby_helper::sorted_keys(rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource* mr)
{
  return cudf::detail::gather(_keys,
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
