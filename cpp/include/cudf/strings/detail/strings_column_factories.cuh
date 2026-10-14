/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Basic type expected for iterators passed to `make_strings_column` that represent string
 * data in device memory.
 */
using string_index_pair = cuda::std::pair<char const*, size_type>;

/**
 * @brief Create a strings-type column from iterators of pointer/size pairs
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param begin First string row (inclusive)
 * @param end Last string row (exclusive)
 * @param stream CUDA stream used for device memory operations
 * @param mr  Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename IndexPairIterator>
std::unique_ptr<column> make_strings_column(IndexPairIterator begin,
                                            IndexPairIterator end,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = cuda::std::distance(begin, end);
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column from the strings sizes
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([] __device__(string_index_pair item) -> size_type {
      return (item.first != nullptr ? static_cast<size_type>(item.second) : size_type{0});
    });
  auto offsets_transformer_itr = cuda::transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  auto new_nulls = cudf::detail::valid_if(begin, end, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask        = (null_count > 0)
                            ? std::move(new_nulls.first)
                            : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);

  // build chars column
  auto chars_data =
    make_chars_buffer(offsets_column->view(), bytes, begin, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

/**
 * @brief Builds the offsets children of a batch of strings columns without synchronizing
 *
 * @tparam OutputType Offset type of the output offsets columns
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param input Per-column iterator to the first string row and number of rows
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return The offsets columns and the device array of per-column chars sizes
 */
template <typename OutputType, typename IndexPairIterator>
std::pair<std::vector<std::unique_ptr<column>>, rmm::device_uvector<int64_t>>
make_offsets_child_column_batch_async(
  std::vector<std::pair<IndexPairIterator, size_type>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_columns = input.size();
  std::vector<std::unique_ptr<column>> offsets_columns(num_columns);
  rmm::device_uvector<int64_t> chars_sizes(num_columns, stream);
  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const string_pairs = input[idx].first;
    auto const string_count = input[idx].second;
    auto offsets            = make_numeric_column(
      data_type{type_to_id<OutputType>()}, string_count + 1, mask_state::UNALLOCATED, stream, mr);

    auto const offsets_transformer = cuda::proclaim_return_type<size_type>(
      [string_count, string_pairs] __device__(size_type idx) -> size_type {
        return idx < string_count ? string_index_pair{string_pairs[idx]}.second : size_type{0};
      });
    auto const input_it  = cudf::detail::make_counting_transform_iterator(0, offsets_transformer);
    auto const d_offsets = offsets->mutable_view().template data<OutputType>();
    auto const output_it = cudf::detail::make_sizes_to_offsets_iterator(
      d_offsets, d_offsets + string_count + 1, chars_sizes.data() + idx);
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           input_it,
                           input_it + string_count + 1,
                           output_it,
                           int64_t{0});
    offsets_columns[idx] = std::move(offsets);
  }

  return {std::move(offsets_columns), std::move(chars_sizes)};
}

/**
 * @brief Creates a batch of strings columns from iterators of pointer/size pairs
 *
 * The output sizes of all columns are synchronized once instead of once per column.
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param input Per-column iterator to the first string row and number of rows
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @param input_consumed Called with a column index once its input is no longer read, so callers
 *        can release the input early and keep the peak memory usage low
 * @return The strings columns, in the order of `input`
 */
template <typename IndexPairIterator>
std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<std::pair<IndexPairIterator, size_type>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::function<void(std::size_t)> const& input_consumed = {})
{
  auto const num_columns = input.size();

  auto [offsets_cols, d_chars_sizes] =
    make_offsets_child_column_batch_async<size_type>(input, stream, mr);

  std::vector<cuda::device_buffer<std::byte>> null_masks;
  null_masks.reserve(num_columns);

  rmm::device_uvector<size_type> d_valid_counts(num_columns, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_valid_counts.begin(),
    d_valid_counts.end(),
    0);

  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const string_pairs = input[idx].first;
    auto const string_count = input[idx].second;
    null_masks.emplace_back(
      cudf::create_null_mask(string_count, mask_state::UNINITIALIZED, stream, mr));

    if (string_count == 0) { continue; }

    constexpr size_type block_size{256};
    auto const grid =
      cudf::detail::grid_1d{static_cast<thread_index_type>(string_count), block_size};
    cudf::detail::valid_if_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        reinterpret_cast<bitmask_type*>(null_masks.back().data()),
        string_pairs,
        string_count,
        [] __device__(string_index_pair const pair) -> bool { return pair.first != nullptr; },
        d_valid_counts.data() + idx);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto const chars_sizes  = cudf::detail::make_std_vector_async(d_chars_sizes, stream);
  auto const valid_counts = cudf::detail::make_std_vector_async(d_valid_counts, stream);

  // Except for other stream syncs in `CUB` that we cannot control,
  // this should be the only stream sync we need in the entire API.
  stream.sync();

  auto const threshold = cudf::strings::get_offset64_threshold();
  auto const overflow_count =
    std::count_if(chars_sizes.begin(), chars_sizes.end(), [threshold](auto const chars_size) {
      return chars_size >= threshold;
    });
  CUDF_EXPECTS(cudf::strings::is_large_strings_enabled() || overflow_count == 0,
               "Size of output exceeds the column size limit",
               std::overflow_error);

  if (overflow_count > 0) {
    std::vector<std::pair<IndexPairIterator, size_type>> long_string_input;
    std::vector<std::size_t> long_string_col_idx;
    long_string_input.reserve(overflow_count);
    long_string_col_idx.reserve(overflow_count);
    for (std::size_t idx = 0; idx < num_columns; ++idx) {
      if (chars_sizes[idx] >= threshold) {
        long_string_input.push_back(input[idx]);
        long_string_col_idx.push_back(idx);
      }
    }

    [[maybe_unused]] auto [new_offsets_cols, d_new_chars_sizes] =
      make_offsets_child_column_batch_async<int64_t>(long_string_input, stream, mr);

    // Update the new offsets columns.
    // The new chars sizes should be the same as before, thus we don't need to update them.
    for (std::size_t idx = 0; idx < long_string_col_idx.size(); ++idx) {
      offsets_cols[long_string_col_idx[idx]] = std::move(new_offsets_cols[idx]);
    }
  }

  std::vector<std::unique_ptr<column>> output(num_columns);
  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const strings_count = input[idx].second;
    if (strings_count == 0) {
      output[idx] = make_empty_column(type_id::STRING);
      if (input_consumed) { input_consumed(idx); }
      continue;
    }

    auto const chars_size  = chars_sizes[idx];
    auto const valid_count = valid_counts[idx];

    auto chars_data = make_chars_buffer(
      offsets_cols[idx]->view(), chars_size, input[idx].first, strings_count, stream, mr);
    if (input_consumed) { input_consumed(idx); }

    auto const null_count = strings_count - valid_count;
    output[idx]           = make_strings_column(
      strings_count,
      std::move(offsets_cols[idx]),
      chars_data.release(),
      null_count,
      null_count ? std::move(null_masks[idx])
                           : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  }

  return output;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
