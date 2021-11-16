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

#include <algorithm>
#include <iterator>
#include <vector>

namespace cudf {
namespace detail {
/**
 * @brief Computes the merger of an array of bitmasks using a binary operator
 *
 * @param op The binary operator used to combine the bitmasks
 * @param destination The bitmask to write result into
 * @param source Array of source mask pointers. All masks must be of same size
 * @param source_begin_bits Array of offsets into corresponding @p source masks.
 *                          Must be same size as source array
 * @param source_size_bits Number of bits in each mask in @p source
 */
template <typename Binop>
__global__ void offset_bitmask_binop(Binop op,
                                     device_span<bitmask_type> destination,
                                     device_span<bitmask_type const*> source,
                                     device_span<size_type const> source_begin_bits,
                                     size_type source_size_bits)
{
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < destination.size();
       destination_word_index += blockDim.x * gridDim.x) {
    bitmask_type destination_word =
      detail::get_mask_offset_word(source[0],
                                   destination_word_index,
                                   source_begin_bits[0],
                                   source_begin_bits[0] + source_size_bits);
    for (size_type i = 1; i < source.size(); i++) {
      destination_word =

        op(destination_word,
           detail::get_mask_offset_word(source[i],
                                        destination_word_index,
                                        source_begin_bits[i],
                                        source_begin_bits[i] + source_size_bits));
    }

    destination[destination_word_index] = destination_word;
  }
}

/**
 * @copydoc bitmask_binop(Binop op, host_span<bitmask_type const *> const, host_span<size_type>
 * const, size_type, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Binop>
rmm::device_buffer bitmask_binop(
  Binop op,
  host_span<bitmask_type const*> masks,
  host_span<size_type const> masks_begin_bits,
  size_type mask_size_bits,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto dest_mask = rmm::device_buffer{bitmask_allocation_size_bytes(mask_size_bits), stream, mr};

  inplace_bitmask_binop(op,
                        device_span<bitmask_type>(static_cast<bitmask_type*>(dest_mask.data()),
                                                  num_bitmask_words(mask_size_bits)),
                        masks,
                        masks_begin_bits,
                        mask_size_bits,
                        stream,
                        mr);

  return dest_mask;
}

/**
 * @brief Performs a merge of the specified bitmasks using the binary operator
 *        provided, and writes in place to destination
 *
 * @param op The binary operator used to combine the bitmasks
 * @param dest_mask Destination to which the merged result is written
 * @param masks The list of data pointers of the bitmasks to be merged
 * @param masks_begin_bits The bit offsets from which each mask is to be merged
 * @param mask_size_bits The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask
 */
template <typename Binop>
void inplace_bitmask_binop(
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

  rmm::device_uvector<bitmask_type const*> d_masks(masks.size(), stream, mr);
  rmm::device_uvector<size_type> d_begin_bits(masks_begin_bits.size(), stream, mr);

  CUDA_TRY(cudaMemcpyAsync(
    d_masks.data(), masks.data(), masks.size_bytes(), cudaMemcpyHostToDevice, stream.value()));
  CUDA_TRY(cudaMemcpyAsync(d_begin_bits.data(),
                           masks_begin_bits.data(),
                           masks_begin_bits.size_bytes(),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  cudf::detail::grid_1d config(dest_mask.size(), 256);
  offset_bitmask_binop<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    op, dest_mask, d_masks, d_begin_bits, mask_size_bits);
  CHECK_CUDA(stream.value());
  stream.synchronize();
}

enum class count_bits_policy : bool {
  SET_BITS,   /// Count set (1) bits
  UNSET_BITS  /// Count unset (0) bits
};

// Count set/unset bits in a segmented null mask, using indices on the device.
rmm::device_uvector<size_type> segmented_count_bits(bitmask_type const* bitmask,
                                                    rmm::device_uvector<size_type> const& d_indices,
                                                    count_bits_policy count_bits,
                                                    rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in every range
 * `[indices_begin[2*i], indices_begin[(2*i)+1])` (where 0 <= i < std::distance(indices_begin,
 * indices_end) / 2).
 *
 * If `bitmask == nullptr`, this function returns a vector containing the
 * segment lengths, or a vector of zeros if `count_unset_bits` is true.
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
 * @param count_bits If SET_BITS, count set (1) bits. If UNSET_BITS, count unset (0) bits
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A vector storing the number of non-zero bits in the specified ranges
 */
template <typename IndexIterator>
std::vector<size_type> segmented_count_bits(bitmask_type const* bitmask,
                                            IndexIterator indices_begin,
                                            IndexIterator indices_end,
                                            count_bits_policy count_bits,
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
    std::vector<size_type> ret(num_indices / 2, 0);
    if (count_bits == count_bits_policy::SET_BITS) {
      for (size_t i = 0; i < num_indices / 2; i++) {
        ret[i] = indices_begin[2 * i + 1] - indices_begin[2 * i];
      }
    }
    return ret;
  }

  // Construct a contiguous host buffer of indices and copy to device.
  auto const h_indices = std::vector<size_type>(indices_begin, indices_end);
  auto const d_indices = make_device_uvector_async(h_indices, stream);

  // Compute the null counts over each segment.
  rmm::device_uvector<size_type> d_null_counts =
    cudf::detail::segmented_count_bits(bitmask, d_indices, count_bits, stream);

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

}  // namespace detail

}  // namespace cudf
