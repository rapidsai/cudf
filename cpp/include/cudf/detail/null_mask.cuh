/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iterator>
#include <optional>
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
 * @param count_ptr Pointer to counter of set bits
 */
template <int block_size, typename Binop>
CUDF_KERNEL void offset_bitmask_binop(Binop op,
                                      device_span<bitmask_type> destination,
                                      device_span<bitmask_type const* const> source,
                                      device_span<size_type const> source_begin_bits,
                                      size_type source_size_bits,
                                      size_type* count_ptr)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();

  auto const last_bit_index  = source_size_bits - 1;
  auto const last_word_index = cudf::word_index(last_bit_index);

  size_type thread_count = 0;

  for (size_type destination_word_index = tid; destination_word_index < destination.size();
       destination_word_index += cudf::detail::grid_1d::grid_stride()) {
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

    if (destination_word_index == last_word_index) {
      // mask out any bits not part of this word
      auto const num_bits_in_last_word = intra_word_index(last_bit_index);
      if (num_bits_in_last_word <
          static_cast<size_type>(detail::size_in_bits<bitmask_type>() - 1)) {
        destination_word &= set_least_significant_bits(num_bits_in_last_word + 1);
      }
    }

    destination[destination_word_index] = destination_word;
    thread_count += __popc(destination_word);
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_count = BlockReduce(temp_storage).Sum(thread_count);

  if (threadIdx.x == 0) { atomicAdd(count_ptr, block_count); }
}

/**
 * @copydoc bitmask_binop(Binop op, host_span<bitmask_type const* const>, host_span<size_type>
 * const, size_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Binop>
std::pair<rmm::device_buffer, size_type> bitmask_binop(Binop op,
                                                       host_span<bitmask_type const* const> masks,
                                                       host_span<size_type const> masks_begin_bits,
                                                       size_type mask_size_bits,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
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
                          stream);

  return std::pair(std::move(dest_mask), null_count);
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
 * @return size_type Count of set bits
 */
template <typename Binop>
size_type inplace_bitmask_binop(Binop op,
                                device_span<bitmask_type> dest_mask,
                                host_span<bitmask_type const* const> masks,
                                host_span<size_type const> masks_begin_bits,
                                size_type mask_size_bits,
                                rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(
    std::all_of(masks_begin_bits.begin(), masks_begin_bits.end(), [](auto b) { return b >= 0; }),
    "Invalid range.");
  CUDF_EXPECTS(mask_size_bits > 0, "Invalid bit range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), [](auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");

  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
  cudf::detail::device_scalar<size_type> d_counter{0, stream, mr};

  auto d_masks      = cudf::detail::make_device_uvector_async(masks, stream, mr);
  auto d_begin_bits = cudf::detail::make_device_uvector_async(masks_begin_bits, stream, mr);

  auto constexpr block_size = 256;
  cudf::detail::grid_1d config(dest_mask.size(), block_size);
  offset_bitmask_binop<block_size>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      op, dest_mask, d_masks, d_begin_bits, mask_size_bits, d_counter.data());
  CUDF_CHECK_CUDA(stream.value());
  return d_counter.value(stream);
}

/**
 * @brief Enum indicating whether to count unset (0) bits or set (1) bits.
 */
enum class count_bits_policy : bool {
  UNSET_BITS,  /// Count unset (0) bits
  SET_BITS     /// Count set (1) bits
};

/**
 * For each range `[first_bit_indices[i], last_bit_indices[i])`
 * (where 0 <= i < `num_ranges`), count the number of bits set outside the range
 * in the boundary words (i.e. words that include either the first or last bit)
 * and subtract the count from the range's null count.
 *
 * Expects `0 <= first_bit_indices[i] <= last_bit_indices[i]`.
 *
 * @param[in] bitmask The bitmask whose non-zero bits outside the range in the
 * boundary words will be counted.
 * @param[in] num_ranges The number of ranges.
 * @param[in] first_bit_indices Random-access input iterator to the sequence of indices (inclusive)
 * of the first bit in each range.
 * @param[in] last_bit_indices Random-access input iterator to the sequence of indices (exclusive)
 * of the last bit in each range.
 * @param[in,out] null_counts Random-access input/output iterator where the number of non-zero bits
 * in each range is updated.
 */
template <typename OffsetIterator, typename OutputIterator>
CUDF_KERNEL void subtract_set_bits_range_boundaries_kernel(bitmask_type const* bitmask,
                                                           size_type num_ranges,
                                                           OffsetIterator first_bit_indices,
                                                           OffsetIterator last_bit_indices,
                                                           OutputIterator null_counts)
{
  constexpr size_type const word_size_in_bits{detail::size_in_bits<bitmask_type>()};

  auto range_id = cudf::detail::grid_1d::global_thread_id();

  while (range_id < num_ranges) {
    size_type const first_bit_index = *(first_bit_indices + range_id);
    size_type const last_bit_index  = *(last_bit_indices + range_id);
    size_type delta                 = 0;

    // Compute delta due to the preceding bits in the first word in the range.
    size_type const first_num_slack_bits = intra_word_index(first_bit_index);
    if (first_num_slack_bits > 0) {
      bitmask_type const word       = bitmask[word_index(first_bit_index)];
      bitmask_type const slack_mask = set_least_significant_bits(first_num_slack_bits);
      delta -= __popc(word & slack_mask);
    }

    // Compute delta due to the following bits in the last word in the range.
    size_type const last_num_slack_bits = (last_bit_index % word_size_in_bits) == 0
                                            ? 0
                                            : word_size_in_bits - intra_word_index(last_bit_index);
    if (last_num_slack_bits > 0) {
      bitmask_type const word       = bitmask[word_index(last_bit_index)];
      bitmask_type const slack_mask = set_most_significant_bits(last_num_slack_bits);
      delta -= __popc(word & slack_mask);
    }

    // Update the null count with the computed delta.
    size_type updated_null_count = *(null_counts + range_id) + delta;
    *(null_counts + range_id)    = updated_null_count;
    range_id += cudf::detail::grid_1d::grid_stride();
  }
}

/**
 * @brief Functor that converts bit segment indices to word segment indices.
 *
 * Converts [first_bit_index, last_bit_index) to [first_word_index,
 * last_word_index). The flag `inclusive` indicates whether the indices are inclusive or exclusive.
 * the end of a segment, in which case the word index should be incremented for
 * bits at the start of a word.
 */
struct bit_to_word_index {
  bit_to_word_index(bool inclusive) : inclusive(inclusive) {}
  __device__ inline size_type operator()(size_type const& bit_index) const
  {
    return word_index(bit_index) + ((inclusive || intra_word_index(bit_index) == 0) ? 0 : 1);
  }
  bool const inclusive;
};

struct popc {
  __device__ inline size_type operator()(bitmask_type word) const { return __popc(word); }
};

// Count set/unset bits in a segmented null mask, using offset iterators accessible by the device.
template <typename OffsetIterator>
rmm::device_uvector<size_type> segmented_count_bits(bitmask_type const* bitmask,
                                                    OffsetIterator first_bit_indices_begin,
                                                    OffsetIterator first_bit_indices_end,
                                                    OffsetIterator last_bit_indices_begin,
                                                    count_bits_policy count_bits,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto const num_ranges =
    static_cast<size_type>(std::distance(first_bit_indices_begin, first_bit_indices_end));
  rmm::device_uvector<size_type> d_bit_counts(num_ranges, stream);

  auto num_set_bits_in_word = thrust::make_transform_iterator(bitmask, popc{});
  auto first_word_indices =
    thrust::make_transform_iterator(first_bit_indices_begin, bit_to_word_index{true});
  auto last_word_indices =
    thrust::make_transform_iterator(last_bit_indices_begin, bit_to_word_index{false});

  // Allocate temporary memory.
  size_t temp_storage_bytes{0};
  CUDF_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(nullptr,
                                                temp_storage_bytes,
                                                num_set_bits_in_word,
                                                d_bit_counts.begin(),
                                                num_ranges,
                                                first_word_indices,
                                                last_word_indices,
                                                stream.value()));
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  // Perform segmented reduction.
  CUDF_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(d_temp_storage.data(),
                                                temp_storage_bytes,
                                                num_set_bits_in_word,
                                                d_bit_counts.begin(),
                                                num_ranges,
                                                first_word_indices,
                                                last_word_indices,
                                                stream.value()));

  // Adjust counts in segment boundaries (if segments are not word-aligned).
  constexpr size_type block_size{256};
  cudf::detail::grid_1d grid(num_ranges, block_size);
  subtract_set_bits_range_boundaries_kernel<<<grid.num_blocks,
                                              grid.num_threads_per_block,
                                              0,
                                              stream.value()>>>(
    bitmask, num_ranges, first_bit_indices_begin, last_bit_indices_begin, d_bit_counts.begin());

  if (count_bits == count_bits_policy::UNSET_BITS) {
    // Convert from set bits counts to unset bits by subtracting the number of
    // set bits from the length of the segment.
    auto segments_begin =
      thrust::make_zip_iterator(first_bit_indices_begin, last_bit_indices_begin);
    auto segment_length_iterator = thrust::transform_iterator(
      segments_begin, cuda::proclaim_return_type<size_type>([] __device__(auto const& segment) {
        auto const begin = thrust::get<0>(segment);
        auto const end   = thrust::get<1>(segment);
        return end - begin;
      }));
    thrust::transform(rmm::exec_policy(stream),
                      segment_length_iterator,
                      segment_length_iterator + num_ranges,
                      d_bit_counts.data(),
                      d_bit_counts.data(),
                      cuda::proclaim_return_type<size_type>(
                        [] __device__(auto segment_size, auto segment_bit_count) {
                          return segment_size - segment_bit_count;
                        }));
  }

  CUDF_CHECK_CUDA(stream.value());
  return d_bit_counts;
}

/**
 * @brief Given two iterators, validate that the iterators represent valid ranges of
 * indices and return the number of ranges.
 *
 * @throws cudf::logic_error if `std::distance(indices_begin, indices_end) % 2 != 0`
 * @throws cudf::logic_error if `indices_begin[2*i] < 0 or indices_begin[2*i] >
 * indices_begin[(2*i)+1]`
 *
 * @param indices_begin An iterator representing the beginning of the ranges of indices
 * @param indices_end An iterator representing the end of the ranges of indices
 *
 * @return The number of segments specified by the input iterators.
 */
template <typename IndexIterator>
size_type validate_segmented_indices(IndexIterator indices_begin, IndexIterator indices_end)
{
  auto const num_indices = static_cast<size_type>(std::distance(indices_begin, indices_end));
  CUDF_EXPECTS(num_indices % 2 == 0,
               "Array of indices needs to have an even number of elements.",
               std::invalid_argument);
  size_type const num_segments = num_indices / 2;
  for (size_type i = 0; i < num_segments; i++) {
    auto begin = indices_begin[2 * i];
    auto end   = indices_begin[2 * i + 1];
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.", std::out_of_range);
    CUDF_EXPECTS(
      end >= begin, "End index cannot be smaller than the starting index.", std::invalid_argument);
  }
  return num_segments;
}

struct index_alternator {
  __device__ inline size_type operator()(size_type const& i) const
  {
    return *(d_indices + 2 * i + (is_end ? 1 : 0));
  }

  bool const is_end = false;
  size_type const* d_indices;
};

/**
 * @brief Given a bitmask, counts the number of set (1) or unset (0) bits in every range
 * `[indices_begin[2*i], indices_begin[(2*i)+1])` (where 0 <= i < std::distance(indices_begin,
 * indices_end) / 2).
 *
 * If `bitmask == nullptr`, this function returns a vector containing the
 * segment lengths, or a vector of zeros if counting unset bits.
 *
 * @throws cudf::logic_error if `bitmask == nullptr`.
 * @throws cudf::logic_error if `std::distance(indices_begin, indices_end) % 2 != 0`.
 * @throws cudf::logic_error if `indices_begin[2*i] < 0 or indices_begin[2*i] >
 * indices_begin[(2*i)+1]`.
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted.
 * @param indices_begin An iterator representing the beginning of the range of indices specifying
 * ranges to count the number of set/unset bits within.
 * @param indices_end An iterator representing the end of the range of indices specifying ranges to
 * count the number of set/unset bits within.
 * @param count_bits If SET_BITS, count set (1) bits. If UNSET_BITS, count unset (0) bits.
 * @param stream CUDA stream used for device memory operations and kernel launches.
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
  CUDF_EXPECTS(bitmask != nullptr, "Invalid bitmask.");
  auto const num_segments = validate_segmented_indices(indices_begin, indices_end);

  // Return an empty vector if there are zero segments.
  if (num_segments == 0) { return std::vector<size_type>{}; }

  // Construct a contiguous host buffer of indices and copy to device.
  auto h_indices = make_empty_host_vector<typename std::iterator_traits<IndexIterator>::value_type>(
    std::distance(indices_begin, indices_end), stream);
  std::copy(indices_begin, indices_end, std::back_inserter(h_indices));
  auto const d_indices =
    make_device_uvector_async(h_indices, stream, cudf::get_current_device_resource_ref());

  // Compute the bit counts over each segment.
  auto first_bit_indices_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), index_alternator{false, d_indices.data()});
  auto const first_bit_indices_end = first_bit_indices_begin + num_segments;
  auto last_bit_indices_begin      = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), index_alternator{true, d_indices.data()});
  rmm::device_uvector<size_type> d_bit_counts =
    cudf::detail::segmented_count_bits(bitmask,
                                       first_bit_indices_begin,
                                       first_bit_indices_end,
                                       last_bit_indices_begin,
                                       count_bits,
                                       stream,
                                       cudf::get_current_device_resource_ref());

  // Copy the results back to the host.
  return make_std_vector_sync(d_bit_counts, stream);
}

// Count non-zero bits in the specified ranges.
template <typename IndexIterator>
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                IndexIterator indices_begin,
                                                IndexIterator indices_end,
                                                rmm::cuda_stream_view stream)
{
  return detail::segmented_count_bits(
    bitmask, indices_begin, indices_end, count_bits_policy::SET_BITS, stream);
}

// Count zero bits in the specified ranges.
template <typename IndexIterator>
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  IndexIterator indices_begin,
                                                  IndexIterator indices_end,
                                                  rmm::cuda_stream_view stream)
{
  return detail::segmented_count_bits(
    bitmask, indices_begin, indices_end, count_bits_policy::UNSET_BITS, stream);
}

// Count valid elements in the specified ranges of a validity bitmask.
template <typename IndexIterator>
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             IndexIterator indices_begin,
                                             IndexIterator indices_end,
                                             rmm::cuda_stream_view stream)
{
  if (bitmask == nullptr) {
    // Return a vector of segment lengths.
    auto const num_segments = validate_segmented_indices(indices_begin, indices_end);
    auto ret                = std::vector<size_type>(num_segments, 0);
    for (size_type i = 0; i < num_segments; i++) {
      ret[i] = indices_begin[2 * i + 1] - indices_begin[2 * i];
    }
    return ret;
  }

  return detail::segmented_count_set_bits(bitmask, indices_begin, indices_end, stream);
}

// Count null elements in the specified ranges of a validity bitmask.
template <typename IndexIterator>
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            IndexIterator indices_begin,
                                            IndexIterator indices_end,
                                            rmm::cuda_stream_view stream)
{
  if (bitmask == nullptr) {
    // Return a vector of zeros.
    auto const num_segments = validate_segmented_indices(indices_begin, indices_end);
    return std::vector<size_type>(num_segments, 0);
  }
  return detail::segmented_count_unset_bits(bitmask, indices_begin, indices_end, stream);
}

/**
 * @brief Create an output null mask whose validity is determined by the
 * validity of any/all elements of segments of an input null mask.
 *
 * @tparam OffsetIterator Random-access input iterator type.
 * @param bitmask Null mask residing in device memory whose segments will be reduced into a new
 * mask.
 * @param first_bit_indices_begin Random-access input iterator to the beginning of a sequence of
 * indices of the first bit in each segment (inclusive).
 * @param first_bit_indices_end Random-access input iterator to the end of a sequence of indices of
 * the first bit in each segment (inclusive).
 * @param last_bit_indices_begin Random-access input iterator to the beginning of a sequence of
 * indices of the last bit in each segment (exclusive).
 * @param null_handling If `null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid. If `null_policy::EXCLUDE`, the reduction is valid if any element in
 * the segment is valid.
 * @param valid_initial_value Indicates whether a valid initial value was provided to the reduction.
 * True indicates a valid initial value, false indicates a null initial value, and null indicates no
 * initial value was provided.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned buffer's device memory.
 * @return A pair containing the reduced null mask and number of nulls.
 */
template <typename OffsetIterator>
std::pair<rmm::device_buffer, size_type> segmented_null_mask_reduction(
  bitmask_type const* bitmask,
  OffsetIterator first_bit_indices_begin,
  OffsetIterator first_bit_indices_end,
  OffsetIterator last_bit_indices_begin,
  null_policy null_handling,
  std::optional<bool> valid_initial_value,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const segments_begin =
    thrust::make_zip_iterator(first_bit_indices_begin, last_bit_indices_begin);
  auto const segment_length_iterator = thrust::make_transform_iterator(
    segments_begin, cuda::proclaim_return_type<size_type>([] __device__(auto const& segment) {
      auto const begin = thrust::get<0>(segment);
      auto const end   = thrust::get<1>(segment);
      return end - begin;
    }));

  auto const num_segments =
    static_cast<size_type>(std::distance(first_bit_indices_begin, first_bit_indices_end));

  if (bitmask == nullptr) {
    return cudf::detail::valid_if(
      segment_length_iterator,
      segment_length_iterator + num_segments,
      [valid_initial_value] __device__(auto const& length) {
        return valid_initial_value.value_or(length > 0);
      },
      stream,
      mr);
  }

  auto const segment_valid_counts =
    cudf::detail::segmented_count_bits(bitmask,
                                       first_bit_indices_begin,
                                       first_bit_indices_end,
                                       last_bit_indices_begin,
                                       cudf::detail::count_bits_policy::SET_BITS,
                                       stream,
                                       cudf::get_current_device_resource_ref());
  auto const length_and_valid_count =
    thrust::make_zip_iterator(segment_length_iterator, segment_valid_counts.begin());
  return cudf::detail::valid_if(
    length_and_valid_count,
    length_and_valid_count + num_segments,
    [null_handling, valid_initial_value] __device__(auto const& length_and_valid_count) {
      auto const length      = thrust::get<0>(length_and_valid_count);
      auto const valid_count = thrust::get<1>(length_and_valid_count);
      return (null_handling == null_policy::EXCLUDE)
               ? (valid_initial_value.value_or(false) || valid_count > 0)
               : (valid_initial_value.value_or(length > 0) && valid_count == length);
    },
    stream,
    mr);
}

}  // namespace detail

}  // namespace cudf
