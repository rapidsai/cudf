/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

namespace cudf {
namespace detail {
/**
 * @brief Computes the merger of an array of bitmasks using a binary operator
 *
 * @tparam block_size Number of threads in each thread block
 * @tparam Binop Type of binary operator
 *
 * @param op The binary operator used to combine the bitmasks
 * @param destination The bitmask to write result into
 * @param source Array of source mask pointers. All masks must be of same size
 * @param source_begin_bits Array of offsets into corresponding @p source masks.
 *                          Must be same size as source array
 * @param source_size_bits Number of bits in each mask in @p source
 * @param count Pointer to counter of set bits
 */
template <int block_size, typename Binop>
__global__ void offset_bitmask_binop(Binop op,
                                     device_span<bitmask_type> destination,
                                     device_span<bitmask_type const*> source,
                                     device_span<size_type const> source_begin_bits,
                                     size_type source_size_bits,
                                     size_type* count_ptr)
{
  constexpr auto const word_size{detail::size_in_bits<bitmask_type>()};
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_type thread_count = 0;

  for (size_type destination_word_index = tid; destination_word_index < destination.size();
       destination_word_index += blockDim.x * gridDim.x) {
    bitmask_type destination_word =
      detail::get_mask_offset_word(source[0],
                                   destination_word_index,
                                   source_begin_bits[0],
                                   source_begin_bits[0] + source_size_bits);
    for (size_type i = 1; i < source.size(); i++) {
      destination_word = op(destination_word,
                            detail::get_mask_offset_word(source[i],
                                                         destination_word_index,
                                                         source_begin_bits[i],
                                                         source_begin_bits[i] + source_size_bits));
    }

    destination[destination_word_index] = destination_word;
    thread_count += __popc(destination_word);
  }

  // Subtract any slack bits from the last word
  if (tid == 0) {
    size_type const last_bit_index = source_size_bits - 1;
    size_type const num_slack_bits = word_size - (last_bit_index % word_size) - 1;
    if (num_slack_bits > 0) {
      size_type const word_index = cudf::word_index(last_bit_index);
      thread_count -= __popc(destination[word_index] & set_most_significant_bits(num_slack_bits));
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_count = BlockReduce(temp_storage).Sum(thread_count);

  if (threadIdx.x == 0) { atomicAdd(count_ptr, block_count); }
}

/**
 * @copydoc bitmask_binop(Binop op, host_span<bitmask_type const *> const, host_span<size_type>
 * const, size_type, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Binop>
std::pair<rmm::device_buffer, size_type> bitmask_binop(
  Binop op,
  host_span<bitmask_type const*> masks,
  host_span<size_type const> masks_begin_bits,
  size_type mask_size_bits,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto dest_mask = rmm::device_buffer{bitmask_allocation_size_bytes(mask_size_bits), stream, mr};
  auto null_count =
    mask_size_bits -
    inplace_bitmask_binop(op,
                          device_span<bitmask_type>(static_cast<bitmask_type*>(dest_mask.data()),
                                                    num_bitmask_words(mask_size_bits)),
                          masks,
                          masks_begin_bits,
                          mask_size_bits,
                          stream,
                          mr);

  return std::make_pair(std::move(dest_mask), null_count);
}

/**
 * @brief Performs a merge of the specified bitmasks using the binary operator
 *        provided, writes in place to destination and returns count of set bits
 *
 * @param[in] op The binary operator used to combine the bitmasks
 * @param[out] dest_mask Destination to which the merged result is written
 * @param[in] masks The list of data pointers of the bitmasks to be merged
 * @param[in] masks_begin_bits The bit offsets from which each mask is to be merged
 * @param[in] mask_size_bits The number of bits to be ANDed in each mask
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned device_buffer
 * @return size_type Count of set bits
 */
template <typename Binop>
size_type inplace_bitmask_binop(
  Binop op,
  device_span<bitmask_type> dest_mask,
  host_span<bitmask_type const*> masks,
  host_span<size_type const> masks_begin_bits,
  size_type mask_size_bits,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(
    std::all_of(masks_begin_bits.begin(), masks_begin_bits.end(), [](auto b) { return b >= 0; }),
    "Invalid range.");
  CUDF_EXPECTS(mask_size_bits > 0, "Invalid bit range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), [](auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");

  rmm::device_scalar<size_type> d_counter{0, stream, mr};
  rmm::device_uvector<bitmask_type const*> d_masks(masks.size(), stream, mr);
  rmm::device_uvector<size_type> d_begin_bits(masks_begin_bits.size(), stream, mr);

  CUDA_TRY(cudaMemcpyAsync(
    d_masks.data(), masks.data(), masks.size_bytes(), cudaMemcpyHostToDevice, stream.value()));
  CUDA_TRY(cudaMemcpyAsync(d_begin_bits.data(),
                           masks_begin_bits.data(),
                           masks_begin_bits.size_bytes(),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  auto constexpr block_size = 256;
  cudf::detail::grid_1d config(dest_mask.size(), block_size);
  offset_bitmask_binop<block_size>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      op, dest_mask, d_masks, d_begin_bits, mask_size_bits, d_counter.data());
  CHECK_CUDA(stream.value());
  return d_counter.value(stream);
}

// Count set bits in a segmented null mask, using indices on the device.
rmm::device_uvector<size_type> segmented_count_set_bits(
  bitmask_type const* bitmask,
  rmm::device_uvector<size_type> const& d_indices,
  rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in every range
 * `[indices_begin[2*i], indices_begin[(2*i)+1])` (where 0 <= i < std::distance(indices_begin,
 * indices_end) / 2).
 *
 * Returns an empty vector if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `std::distance(indices_begin, indices_end) % 2 != 0`
 * @throws cudf::logic_error if `indices_begin[2*i] < 0 or indices_begin[2*i] >
 * indices_begin[(2*i)+1]`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param indices_begin An iterator representing the beginning of the range of indices specifying
 * ranges to count the number of set bits within
 * @param indices_end An iterator representing the end of the range of indices specifying ranges to
 * count the number of set bits within
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A vector storing the number of non-zero bits in the specified ranges
 */
template <typename IndexIterator>
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                IndexIterator indices_begin,
                                                IndexIterator indices_end,
                                                rmm::cuda_stream_view stream)
{
  size_t const num_indices = std::distance(indices_begin, indices_end);

  CUDF_EXPECTS(num_indices % 2 == 0, "Array of indices needs to have an even number of elements.");
  for (size_t i = 0; i < num_indices / 2; i++) {
    auto begin = indices_begin[i * 2];
    auto end   = indices_begin[i * 2 + 1];
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
    CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
  }

  if (num_indices == 0) {
    return std::vector<size_type>{};
  } else if (bitmask == nullptr) {
    std::vector<size_type> ret(num_indices / 2);
    for (size_t i = 0; i < num_indices / 2; i++) {
      ret[i] = indices_begin[2 * i + 1] - indices_begin[2 * i];
    }
    return ret;
  }

  // Construct a contiguous host buffer of indices and copy to device.
  auto const h_indices = std::vector<size_type>(indices_begin, indices_end);
  auto const d_indices = make_device_uvector_async(h_indices, stream);

  // Compute the null counts over each segment.
  rmm::device_uvector<size_type> d_null_counts =
    segmented_count_set_bits(bitmask, d_indices, stream);

  // Copy the results back to the host.
  size_type const num_ranges = num_indices / 2;
  std::vector<size_type> ret(num_ranges);
  CUDA_TRY(cudaMemcpyAsync(ret.data(),
                           d_null_counts.data(),
                           num_ranges * sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));

  stream.synchronize();  // now ret is valid.

  return ret;
}

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in every range
 * `[indices_begin[2*i], indices_begin[(2*i)+1])` (where 0 <= i < std::distance(indices_begin,
 * indices_end) / 2).
 *
 * Returns an empty vector if `bitmask == nullptr`.
 *
 * @throws cudf::logic_error if `std::distance(indices_begin, indices_end) % 2 != 0`
 * @throws cudf::logic_error if `indices_begin[2*i] < 0 or indices_begin[2*i] >
 * indices_begin[(2*i)+1]`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted
 * @param indices_begin An iterator representing the beginning of the range of indices specifying
 * ranges to count the number of unset bits within
 * @param indices_end An iterator representing the end of the range of indices specifying ranges to
 * count the number of unset bits within
 * @param streaam CUDA stream used for device memory operations and kernel launches
 *
 * @return A vector storing the number of non-zero bits in the specified ranges
 */
template <typename IndexIterator>
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  IndexIterator indices_begin,
                                                  IndexIterator indices_end,
                                                  rmm::cuda_stream_view stream)
{
  size_t const num_indices = std::distance(indices_begin, indices_end);

  if (num_indices == 0) {
    return std::vector<size_type>{};
  } else if (bitmask == nullptr) {
    return std::vector<size_type>(num_indices / 2, 0);
  }

  auto ret = segmented_count_set_bits(bitmask, indices_begin, indices_end, stream);
  for (size_t i = 0; i < ret.size(); i++) {
    auto begin = indices_begin[i * 2];
    auto end   = indices_begin[i * 2 + 1];
    ret[i]     = (end - begin) - ret[i];
  }

  return ret;
}

}  // namespace detail

}  // namespace cudf
