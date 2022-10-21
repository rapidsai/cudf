/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief The enum specifying which sorting method to use (stable or unstable).
 */
enum class sort_method { STABLE, UNSTABLE };

/**
 * @brief Functor performs faster sort on eligible columns
 */
struct column_fast_sort_fn {
  /**
   * @brief Run time check for faster sort eligible column
   */
  static bool is_fast_sort_supported(column_view const& col)
  {
    return !col.has_nulls() and cudf::is_numeric(col.type()) and
           not cudf::is_floating_point(col.type()) and cudf::is_relationally_comparable(col.type());
  }

  /**
   * @brief Compile time check for allowing radix sort for column type.
   */
  template <typename T>
  static constexpr bool is_radix_sort_supported()
  {
    return std::is_integral<T>();
  }

  template <typename KeyT, typename ValueT, typename OffsetIteratorT>
  void radix_sort_ascending(KeyT const* keys_in,
                            KeyT* keys_out,
                            ValueT const* values_in,
                            ValueT* values_out,
                            int num_items,
                            int num_segments,
                            OffsetIteratorT begin_offsets,
                            OffsetIteratorT end_offsets,
                            rmm::cuda_stream_view stream)
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             keys_in,
                                             keys_out,
                                             values_in,
                                             values_out,
                                             num_items,
                                             num_segments,
                                             begin_offsets,
                                             end_offsets,
                                             0,
                                             sizeof(KeyT) * 8,
                                             stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             keys_in,
                                             keys_out,
                                             values_in,
                                             values_out,
                                             num_items,
                                             num_segments,
                                             begin_offsets,
                                             end_offsets,
                                             0,
                                             sizeof(KeyT) * 8,
                                             stream.value());
  }

  template <typename KeyT, typename ValueT, typename OffsetIteratorT>
  void radix_sort_descending(KeyT const* keys_in,
                             KeyT* keys_out,
                             ValueT const* values_in,
                             ValueT* values_out,
                             int num_items,
                             int num_segments,
                             OffsetIteratorT begin_offsets,
                             OffsetIteratorT end_offsets,
                             rmm::cuda_stream_view stream)
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage.data(),
                                                       temp_storage_bytes,
                                                       keys_in,
                                                       keys_out,
                                                       values_in,
                                                       values_out,
                                                       num_items,
                                                       num_segments,
                                                       begin_offsets,
                                                       end_offsets,
                                                       0,
                                                       sizeof(KeyT) * 8,
                                                       stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage.data(),
                                                       temp_storage_bytes,
                                                       keys_in,
                                                       keys_out,
                                                       values_in,
                                                       values_out,
                                                       num_items,
                                                       num_segments,
                                                       begin_offsets,
                                                       end_offsets,
                                                       0,
                                                       sizeof(KeyT) * 8,
                                                       stream.value());
  }

  template <typename T>
  void radix_sort(column_view const& input,
                  column_view const& segment_offsets,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    auto temp_col =
      cudf::detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, stream);
    mutable_column_view output_view = temp_col->mutable_view();

    if (ascending) {
      radix_sort_ascending(input.begin<T>(),
                           output_view.begin<T>(),
                           indices.begin<size_type>(),
                           indices.begin<size_type>(),
                           input.size(),
                           segment_offsets.size() - 1,
                           segment_offsets.begin<size_type>(),
                           segment_offsets.begin<size_type>() + 1,
                           stream);
    } else {
      radix_sort_descending(input.begin<T>(),
                            output_view.begin<T>(),
                            indices.begin<size_type>(),
                            indices.begin<size_type>(),
                            input.size(),
                            segment_offsets.size() - 1,
                            segment_offsets.begin<size_type>(),
                            segment_offsets.begin<size_type>() + 1,
                            stream);
    }
  }

  template <typename T, std::enable_if_t<is_radix_sort_supported<T>()>* = nullptr>
  void operator()(column_view const& input,
                  column_view const& segment_offsets,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    radix_sort<T>(input, segment_offsets, indices, ascending, stream);
  }

  template <typename T, std::enable_if_t<!is_radix_sort_supported<T>()>* = nullptr>
  void operator()(
    column_view const&, column_view const&, mutable_column_view&, bool, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Column type is not fast sortable");
  }
};

/**
 * @brief Performs faster sort on eligible columns
 *
 * Check the `is_fast_sort_supported()==true` on the input column before using this function.
 *
 * @param input Column to sort
 * @param segment_offsets Identifies segments to sort within
 * @param column_order Sort ascending or descending
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> fast_segmented_sorted_order(column_view const& input,
                                                    column_view const& segment_offsets,
                                                    order const& column_order,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  auto sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view indices_view = sorted_indices->mutable_view();
  thrust::sequence(
    rmm::exec_policy(stream), indices_view.begin<size_type>(), indices_view.end<size_type>(), 0);
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               column_fast_sort_fn{},
                                               input,
                                               segment_offsets,
                                               indices_view,
                                               column_order == order::ASCENDING,
                                               stream);
  return sorted_indices;
}

/**
 * @brief Builds indices to identify segments to sort
 *
 * The segments are added to the input table-view keys so they
 * are lexicographically sorted within the segmented groups.
 *
 * ```
 * Example 1:
 * num_rows = 10
 * offsets = {0, 3, 7, 10}
 * segment-indices -> { 3,3,3, 7,7,7,7, 10,10,10 }
 * ```
 *
 * ```
 * Example 2: (offsets do not cover all indices)
 * num_rows = 10
 * offsets = {3, 7}
 * segment-indices -> { 0,1,2, 7,7,7,7, 8,9,10 }
 * ```
 *
 * @param num_rows Total number of rows in the input keys to sort
 * @param offsets The offsets identifying the segments
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> segment_ids(num_rows, stream);

  auto offset_begin  = offsets.begin<size_type>();
  auto offset_end    = offsets.end<size_type>();
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(rmm::exec_policy(stream),
                    counting_iter,
                    counting_iter + segment_ids.size(),
                    segment_ids.begin(),
                    [offset_begin, offset_end] __device__(auto idx) {
                      if (offset_begin == offset_end || idx < *offset_begin) { return idx; }
                      if (idx >= *(offset_end - 1)) { return idx + 1; }
                      return static_cast<size_type>(
                        *thrust::upper_bound(thrust::seq, offset_begin, offset_end, idx));
                    });
  return segment_ids;
}

std::unique_ptr<column> segmented_sorted_order_common(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  sort_method sorting,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(segment_offsets.type() == data_type(type_to_id<size_type>()),
               "segment offsets should be size_type");

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(keys.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(keys.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null_precedence size.");
  }

  // the average list size at which to prefer fast sort
  constexpr cudf::size_type MIN_AVG_LIST_SIZE_FOR_FAST_SORT{100};

  // fast-path for single column sort
  if (keys.num_columns() == 1 and sorting == sort_method::UNSTABLE and
      column_fast_sort_fn::is_fast_sort_supported(keys.column(0)) and
      (segment_offsets.size() > 0) and
      ((keys.column(0).size() / segment_offsets.size()) > MIN_AVG_LIST_SIZE_FOR_FAST_SORT)) {
    auto const col_order = column_order.empty() ? order::ASCENDING : column_order.front();
    return fast_segmented_sorted_order(keys.column(0), segment_offsets, col_order, stream, mr);
  }

  // Get segment id of each element in all segments.
  auto segment_ids = get_segment_indices(keys.num_rows(), segment_offsets, stream);

  // insert segment id before all columns.
  std::vector<column_view> keys_with_segid;
  keys_with_segid.reserve(keys.num_columns() + 1);
  keys_with_segid.push_back(
    column_view(data_type(type_to_id<size_type>()), segment_ids.size(), segment_ids.data()));
  keys_with_segid.insert(keys_with_segid.end(), keys.begin(), keys.end());
  auto segid_keys = table_view(keys_with_segid);

  auto prepend_default = [](auto const& vector, auto default_value) {
    if (vector.empty()) return vector;
    std::remove_cv_t<std::remove_reference_t<decltype(vector)>> pre_vector;
    pre_vector.reserve(pre_vector.size() + 1);
    pre_vector.push_back(default_value);
    pre_vector.insert(pre_vector.end(), vector.begin(), vector.end());
    return pre_vector;
  };
  auto child_column_order    = prepend_default(column_order, order::ASCENDING);
  auto child_null_precedence = prepend_default(null_precedence, null_order::AFTER);

  // return sorted order of child columns
  return sorting == sort_method::STABLE
           ? detail::stable_sorted_order(
               segid_keys, child_column_order, child_null_precedence, stream, mr)
           : detail::sorted_order(
               segid_keys, child_column_order, child_null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key_common(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    sort_method sorting,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");
  auto sorted_order = sorting == sort_method::STABLE
                        ? stable_segmented_sorted_order(keys,
                                                        segment_offsets,
                                                        column_order,
                                                        null_precedence,
                                                        stream,
                                                        rmm::mr::get_current_device_resource())
                        : segmented_sorted_order(keys,
                                                 segment_offsets,
                                                 column_order,
                                                 null_precedence,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());

  // Gather segmented sort of child value columns`
  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common(
    keys, segment_offsets, column_order, null_precedence, sort_method::UNSTABLE, stream, mr);
}

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common(
    keys, segment_offsets, column_order, null_precedence, sort_method::STABLE, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common(values,
                                      keys,
                                      segment_offsets,
                                      column_order,
                                      null_precedence,
                                      sort_method::UNSTABLE,
                                      stream,
                                      mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common(
    values, keys, segment_offsets, column_order, null_precedence, sort_method::STABLE, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

}  // namespace cudf
