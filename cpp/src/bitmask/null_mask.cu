/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace cudf {
size_type state_null_count(mask_state state, size_type size)
{
  switch (state) {
    case mask_state::UNALLOCATED: return 0;
    case mask_state::ALL_NULL: return size;
    case mask_state::ALL_VALID: return 0;
    default: CUDF_FAIL("Invalid null mask state.", std::invalid_argument);
  }
}

// Computes required allocation size of a bitmask
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits, std::size_t padding_boundary)
{
  CUDF_EXPECTS(padding_boundary > 0, "Invalid padding boundary");
  auto necessary_bytes = cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes = padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                                           necessary_bytes, padding_boundary);
  return padded_bytes;
}

// Computes number of *actual* bitmask_type elements needed
size_type num_bitmask_words(size_type number_of_bits)
{
  return cudf::util::div_rounding_up_safe<size_type>(number_of_bits,
                                                     detail::size_in_bits<bitmask_type>());
}

namespace detail {

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  size_type mask_size{0};

  if (state != mask_state::UNALLOCATED) { mask_size = bitmask_allocation_size_bytes(size); }

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != mask_state::UNINITIALIZED) {
    uint8_t fill_value = (state == mask_state::ALL_VALID) ? 0xff : 0x00;
    CUDF_CUDA_TRY(cudaMemsetAsync(
      static_cast<bitmask_type*>(mask.data()), fill_value, mask_size, stream.value()));
  }

  return mask;
}

namespace {
CUDF_KERNEL void set_null_mask_kernel(bitmask_type* __restrict__ destination,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      bool valid,
                                      size_type number_of_mask_words)
{
  auto x                            = destination + word_index(begin_bit);
  thread_index_type const last_word = word_index(end_bit) - word_index(begin_bit);
  bitmask_type fill_value           = valid ? 0xffff'ffff : 0;

  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (thread_index_type destination_word_index = grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    if (destination_word_index == 0 || destination_word_index == last_word) {
      bitmask_type mask = ~bitmask_type{0};
      if (destination_word_index == 0) {
        mask = ~(set_least_significant_bits(intra_word_index(begin_bit)));
      }
      if (destination_word_index == last_word) {
        mask = mask & set_least_significant_bits(intra_word_index(end_bit));
      }
      x[destination_word_index] =
        valid ? x[destination_word_index] | mask : x[destination_word_index] & ~mask;
    } else {
      x[destination_word_index] = fill_value;
    }
  }
}
}  // namespace

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit <= end_bit, "Invalid bit range.");
  if (begin_bit == end_bit) return;
  if (bitmask != nullptr) {
    auto number_of_mask_words =
      num_bitmask_words(end_bit) - begin_bit / detail::size_in_bits<bitmask_type>();
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    set_null_mask_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      static_cast<bitmask_type*>(bitmask), begin_bit, end_bit, valid, number_of_mask_words);
    CUDF_CHECK_CUDA(stream.value());
  }
}

}  // namespace detail

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return detail::create_null_mask(size, state, stream, mr);
}

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream)
{
  return detail::set_null_mask(bitmask, begin_bit, end_bit, valid, stream);
}

namespace detail {

namespace {
/**
 * @brief Copies the bits starting at the specified offset from a source
 * bitmask into the destination bitmask.
 *
 * Bit `i` in `destination` will be equal to bit `i + offset` from `source`.
 *
 * @param destination The mask to copy into
 * @param source The mask to copy from
 * @param source_begin_bit The offset into `source` from which to begin the copy
 * @param source_end_bit   The offset into `source` till which copying is done
 * @param number_of_mask_words The number of `cudf::bitmask_type` words to copy
 */
// TODO: Also make binops test that uses offset in column_view
CUDF_KERNEL void copy_offset_bitmask(bitmask_type* __restrict__ destination,
                                     bitmask_type const* __restrict__ source,
                                     size_type source_begin_bit,
                                     size_type source_end_bit,
                                     size_type number_of_mask_words)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (thread_index_type destination_word_index = grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    destination[destination_word_index] = detail::get_mask_offset_word(
      source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

}  // namespace

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit <= end_bit, "Invalid bit range.");
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(end_bit - begin_bit);
  if ((mask == nullptr) || (num_bytes == 0)) { return dest_mask; }
  if (begin_bit == 0) {
    dest_mask = rmm::device_buffer{static_cast<void const*>(mask), num_bytes, stream, mr};
  } else {
    auto number_of_mask_words = num_bitmask_words(end_bit - begin_bit);
    dest_mask                 = rmm::device_buffer{num_bytes, stream, mr};
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      static_cast<bitmask_type*>(dest_mask.data()), mask, begin_bit, end_bit, number_of_mask_words);
    CUDF_CHECK_CUDA(stream.value());
  }
  return dest_mask;
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const& view,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.nullable()) {
    null_mask =
      copy_bitmask(view.null_mask(), view.offset(), view.offset() + view.size(), stream, mr);
  }
  return null_mask;
}

namespace {
/**
 * @brief Counts the number of non-zero bits in a bitmask in the range
 * `[first_bit_index, last_bit_index]`.
 *
 * Expects `0 <= first_bit_index <= last_bit_index`.
 *
 * @param[in] bitmask The bitmask whose non-zero bits will be counted.
 * @param[in] first_bit_index The index (inclusive) of the first bit to count
 * @param[in] last_bit_index The index (inclusive) of the last bit to count
 * @param[out] global_count The number of non-zero bits in the specified range
 */
template <size_type block_size>
CUDF_KERNEL void count_set_bits_kernel(bitmask_type const* bitmask,
                                       size_type first_bit_index,
                                       size_type last_bit_index,
                                       size_type* global_count)
{
  constexpr auto const word_size{detail::size_in_bits<bitmask_type>()};

  auto const first_word_index{word_index(first_bit_index)};
  auto const last_word_index{word_index(last_bit_index)};
  thread_index_type const tid         = grid_1d::global_thread_id<block_size>();
  thread_index_type const stride      = grid_1d::grid_stride<block_size>();
  thread_index_type thread_word_index = tid + first_word_index;
  size_type thread_count{0};

  // First, just count the bits in all words
  while (thread_word_index <= last_word_index) {
    thread_count += __popc(bitmask[thread_word_index]);
    thread_word_index += stride;
  }

  // Subtract any slack bits counted from the first and last word
  // Two threads handle this -- one for first word, one for last
  if (tid < 2) {
    bool const first{tid == 0};
    bool const last{not first};

    size_type bit_index  = (first) ? first_bit_index : last_bit_index;
    size_type word_index = (first) ? first_word_index : last_word_index;

    size_type num_slack_bits = bit_index % word_size;
    if (last) { num_slack_bits = word_size - num_slack_bits - 1; }

    if (num_slack_bits > 0) {
      bitmask_type word = bitmask[word_index];
      auto slack_mask   = (first) ? set_least_significant_bits(num_slack_bits)
                                  : set_most_significant_bits(num_slack_bits);

      thread_count -= __popc(word & slack_mask);
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_count{BlockReduce(temp_storage).Sum(thread_count)};

  if (threadIdx.x == 0) { atomicAdd(global_count, block_count); }
}

}  // namespace

// Count non-zero bits in the specified range
cudf::size_type count_set_bits(bitmask_type const* bitmask,
                               size_type start,
                               size_type stop,
                               rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(bitmask != nullptr, "Invalid bitmask.");
  CUDF_EXPECTS(start >= 0, "Invalid range.");
  CUDF_EXPECTS(start <= stop, "Invalid bit range.");

  auto const num_bits_to_count = stop - start;
  if (num_bits_to_count == 0) { return 0; }

  auto const num_words = num_bitmask_words(num_bits_to_count);

  constexpr size_type block_size{256};

  cudf::detail::grid_1d grid(num_words, block_size);

  cudf::detail::device_scalar<size_type> non_zero_count(0, stream);

  count_set_bits_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      bitmask, start, stop - 1, non_zero_count.data());

  return non_zero_count.value(stream);
}

// Count zero bits in the specified range
cudf::size_type count_unset_bits(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 rmm::cuda_stream_view stream)
{
  auto const num_set_bits   = detail::count_set_bits(bitmask, start, stop, stream);
  auto const total_num_bits = (stop - start);
  return total_num_bits - num_set_bits;
}

// Count valid elements in the specified range of a validity bitmask
cudf::size_type valid_count(bitmask_type const* bitmask,
                            size_type start,
                            size_type stop,
                            rmm::cuda_stream_view stream)
{
  if (bitmask == nullptr) {
    CUDF_EXPECTS(start >= 0, "Invalid range.");
    CUDF_EXPECTS(start <= stop, "Invalid bit range.");
    auto const total_num_bits = (stop - start);
    return total_num_bits;
  }

  return detail::count_set_bits(bitmask, start, stop, stream);
}

// Count null elements in the specified range of a validity bitmask
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           rmm::cuda_stream_view stream)
{
  if (bitmask == nullptr) {
    CUDF_EXPECTS(start >= 0, "Invalid range.");
    CUDF_EXPECTS(start <= stop, "Invalid bit range.");
    return 0;
  }

  return detail::count_unset_bits(bitmask, start, stop, stream);
}

// Count non-zero bits in the specified ranges of a bitmask
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                host_span<size_type const> indices,
                                                rmm::cuda_stream_view stream)
{
  return detail::segmented_count_set_bits(bitmask, indices.begin(), indices.end(), stream);
}

// Count zero bits in the specified ranges of a bitmask
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  host_span<size_type const> indices,
                                                  rmm::cuda_stream_view stream)
{
  return detail::segmented_count_unset_bits(bitmask, indices.begin(), indices.end(), stream);
}

// Count valid elements in the specified ranges of a validity bitmask
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             host_span<size_type const> indices,
                                             rmm::cuda_stream_view stream)
{
  return detail::segmented_valid_count(bitmask, indices.begin(), indices.end(), stream);
}

// Count null elements in the specified ranges of a validity bitmask
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            host_span<size_type const> indices,
                                            rmm::cuda_stream_view stream)
{
  return detail::segmented_null_count(bitmask, indices.begin(), indices.end(), stream);
}

// Inplace Bitwise AND of the masks
cudf::size_type inplace_bitmask_and(device_span<bitmask_type> dest_mask,
                                    host_span<bitmask_type const* const> masks,
                                    host_span<size_type const> begin_bits,
                                    size_type mask_size,
                                    rmm::cuda_stream_view stream)
{
  return inplace_bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    dest_mask,
    masks,
    begin_bits,
    mask_size,
    stream);
}

// Bitwise AND of the masks
std::pair<rmm::device_buffer, size_type> bitmask_and(host_span<bitmask_type const* const> masks,
                                                     host_span<size_type const> begin_bits,
                                                     size_type mask_size,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  return bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    masks,
    begin_bits,
    mask_size,
    stream,
    mr);
}

// Returns the bitwise AND of the null masks of all columns in the table view
std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.num_rows() == 0 or view.num_columns() == 0) {
    return std::pair(std::move(null_mask), 0);
  }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> offsets;
  for (auto&& col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }

  if (masks.size() > 0) {
    return cudf::detail::bitmask_binop(
      [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
      masks,
      offsets,
      view.num_rows(),
      stream,
      mr);
  }

  return std::pair(std::move(null_mask), 0);
}

// Returns the bitwise OR of the null masks of all columns in the table view
std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.num_rows() == 0 or view.num_columns() == 0) {
    return std::pair(std::move(null_mask), 0);
  }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> offsets;
  for (auto&& col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }

  if (static_cast<size_type>(masks.size()) == view.num_columns()) {
    return cudf::detail::bitmask_binop(
      [] __device__(bitmask_type left, bitmask_type right) { return left | right; },
      masks,
      offsets,
      view.num_rows(),
      stream,
      mr);
  }

  return std::pair(std::move(null_mask), 0);
}

void set_all_valid_null_masks(column_view const& input,
                              column& output,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  if (input.nullable()) {
    auto mask = detail::create_null_mask(output.size(), mask_state::ALL_VALID, stream, mr);
    output.set_null_mask(std::move(mask), 0);

    for (size_type i = 0; i < input.num_children(); ++i) {
      set_all_valid_null_masks(input.child(i), output.child(i), stream, mr);
    }
  }
}

}  // namespace detail

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_bitmask(mask, begin_bit, end_bit, stream, mr);
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const& view,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_bitmask(view, stream, mr);
}

std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bitmask_and(view, stream, mr);
}

std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bitmask_or(view, stream, mr);
}

// Count non-zero bits in the specified range
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::null_count(bitmask, start, stop, stream);
}

}  // namespace cudf
