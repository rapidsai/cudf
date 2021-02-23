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
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

using cudf::detail::device_span;

namespace cudf {
namespace {
/**
 * @brief Computes the merger of an array of bitmasks using a binary operator
 *
 * @param op The binary operator used to combine the bitmasks
 * @param destination The bitmask to write result into
 * @param source Array of source mask pointers. All masks must be of same size
 * @param begin_bit Array of offsets into corresponding @p source masks.
 *                  Must be same size as source array
 * @param num_sources Number of masks in @p source array
 * @param source_size Number of bits in each mask in @p source
 * @param number_of_mask_words The number of words of type bitmask_type to copy
 */
template <typename Binop>
__global__ void offset_bitmask_binop(Binop op,
                                     device_span<bitmask_type> destination,
                                     device_span<bitmask_type const *> source,
                                     device_span<size_type> begin_bit,
                                     size_type source_size)
{
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < destination.size();
       destination_word_index += blockDim.x * gridDim.x) {
    bitmask_type destination_word = detail::get_mask_offset_word(
      source[0], destination_word_index, begin_bit[0], begin_bit[0] + source_size);
    for (size_type i = 1; i < source.size(); i++) {
      destination_word =
        op(destination_word,
           detail::get_mask_offset_word(
             source[i], destination_word_index, begin_bit[i], begin_bit[i] + source_size));
    }

    destination[destination_word_index] = destination_word;
  }
}
}  // namespace
namespace detail {
/**
 * @copydoc bitmask_binop(Binop op, std::vector<bitmask_type const*>, std::vector<size_type> const&,
 * size_type, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Binop>
rmm::device_buffer bitmask_binop(
  Binop op,
  std::vector<bitmask_type const *> const &masks,
  std::vector<size_type> const &begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(std::all_of(begin_bits.begin(), begin_bits.end(), [](auto b) { return b >= 0; }),
               "Invalid range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), [](auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(mask_size);

  rmm::device_vector<bitmask_type const *> d_masks(masks);
  rmm::device_vector<size_type> d_begin_bits(begin_bits);

  dest_mask = rmm::device_buffer{num_bytes, stream, mr};

  inplace_bitmask_binop(op,
                        device_span<bitmask_type>(static_cast<bitmask_type *>(dest_mask.data()),
                                                  num_bitmask_words(mask_size)),
                        device_span<bitmask_type const *>(d_masks.data().get(), d_masks.size()),
                        device_span<size_type>(d_begin_bits.data().get(), d_begin_bits.size()),
                        mask_size,
                        stream,
                        mr);

  return dest_mask;
}

/**
 * @brief Performs a merger of the specified bitmasks using the binary operator
 *        provided, and writes in place to destination
 *
 * @param op The binary operator used to combine the bitmasks
 * @param dest_mask Destination to which the AND result is written
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask
 */
template <typename Binop>
void inplace_bitmask_binop(
  Binop op,
  device_span<bitmask_type> dest_mask,
  device_span<bitmask_type const *> masks,
  device_span<size_type> begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(mask_size > 0, "Invalid bit range.");

  cudf::detail::grid_1d config(dest_mask.size(), 256);
  offset_bitmask_binop<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    op, dest_mask, masks, begin_bits, mask_size);
  CHECK_CUDA(stream.value());
}

}  // namespace detail

}  // namespace cudf
