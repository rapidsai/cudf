/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace cudf {
size_type state_null_count(mask_state state, size_type size)
{
  switch (state) {
    case mask_state::UNALLOCATED: return 0;
    case mask_state::UNINITIALIZED: return UNKNOWN_NULL_COUNT;
    case mask_state::ALL_NULL: return size;
    case mask_state::ALL_VALID: return 0;
    default: CUDF_FAIL("Invalid null mask state.");
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
                                    rmm::mr::device_memory_resource *mr)
{
  size_type mask_size{0};

  if (state != mask_state::UNALLOCATED) { mask_size = bitmask_allocation_size_bytes(size); }

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != mask_state::UNINITIALIZED) {
    uint8_t fill_value = (state == mask_state::ALL_VALID) ? 0xff : 0x00;
    CUDA_TRY(cudaMemsetAsync(
      static_cast<bitmask_type *>(mask.data()), fill_value, mask_size, stream.value()));
  }

  return mask;
}

namespace {
__global__ void set_null_mask_kernel(bitmask_type *__restrict__ destination,
                                     size_type begin_bit,
                                     size_type end_bit,
                                     bool valid,
                                     size_type number_of_mask_words)
{
  auto x                  = destination + word_index(begin_bit);
  const auto last_word    = word_index(end_bit) - word_index(begin_bit);
  bitmask_type fill_value = (valid == true) ? 0xffffffff : 0x00;

  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    if (destination_word_index == 0 || destination_word_index == last_word) {
      bitmask_type mask = ~bitmask_type{0};
      if (destination_word_index == 0) {
        mask = ~(set_least_significant_bits(intra_word_index(begin_bit)));
      }
      if (destination_word_index == last_word) {
        mask = mask & set_least_significant_bits(intra_word_index(end_bit));
      }
      x[destination_word_index] =
        (valid == true) ? x[destination_word_index] | mask : x[destination_word_index] & ~mask;
    } else {
      x[destination_word_index] = fill_value;
    }
  }
}
}  // namespace

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type *bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit < end_bit, "Invalid bit range.");
  if (bitmask != nullptr) {
    auto number_of_mask_words =
      num_bitmask_words(end_bit) - begin_bit / detail::size_in_bits<bitmask_type>();
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    set_null_mask_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      static_cast<bitmask_type *>(bitmask), begin_bit, end_bit, valid, number_of_mask_words);
    CHECK_CUDA(stream.value());
  }
}

}  // namespace detail

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::create_null_mask(size, state, rmm::cuda_stream_default, mr);
}

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type *bitmask, size_type begin_bit, size_type end_bit, bool valid)
{
  return detail::set_null_mask(bitmask, begin_bit, end_bit, valid);
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
__global__ void count_set_bits_kernel(bitmask_type const *bitmask,
                                      size_type first_bit_index,
                                      size_type last_bit_index,
                                      size_type *global_count)
{
  constexpr auto const word_size{detail::size_in_bits<bitmask_type>()};

  auto const first_word_index{word_index(first_bit_index)};
  auto const last_word_index{word_index(last_bit_index)};
  auto const tid         = threadIdx.x + blockIdx.x * blockDim.x;
  auto thread_word_index = tid + first_word_index;
  size_type thread_count{0};

  // First, just count the bits in all words
  while (thread_word_index <= last_word_index) {
    thread_count += __popc(bitmask[thread_word_index]);
    thread_word_index += blockDim.x * gridDim.x;
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

/**
 * For each range `[first_bit_indices[i], last_bit_indices[i])`
 * (where 0 <= i < `num_ranges`), count the number of bits set outside the range
 * in the boundary words (i.e. words that include either
 * `first_bit_indices[i]'th` bit or `(last_bit_indices[i] - 1)'th` bit) and
 * subtract the count from the range's null count.
 *
 * Expects `0 <= first_bit_indices[i] <= last_bit_indices[i]`.
 *
 * @param[in] bitmask The bitmask whose non-zero bits outside the range in the
 * boundary words will be counted.
 * @param[in] num_ranges The number of ranges
 * @param[in] first_bit_indices The indices (inclusive) of the first bit in each
 * range
 * @param[in] last_bit_indices The indices (exclusive) of the last bit in each
 * range
 * @param[in,out] null_counts The number of non-zero bits in each range to be
 * updated
 */
template <typename OffsetIterator, typename OutputIterator>
__global__ void subtract_set_bits_range_boundaries_kerenel(bitmask_type const *bitmask,
                                                           size_type num_ranges,
                                                           OffsetIterator first_bit_indices,
                                                           OffsetIterator last_bit_indices,
                                                           OutputIterator null_counts)
{
  constexpr size_type const word_size_in_bits{detail::size_in_bits<bitmask_type>()};

  cudf::size_type const tid = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type range_id  = tid;

  while (range_id < num_ranges) {
    size_type const first_bit_index = *(first_bit_indices + range_id);
    size_type const last_bit_index  = *(last_bit_indices + range_id);
    size_type delta                 = 0;
    size_type num_slack_bits        = 0;

    // compute delta due to the preceding bits in the first word in the range

    num_slack_bits = intra_word_index(first_bit_index);
    if (num_slack_bits > 0) {
      bitmask_type word       = bitmask[word_index(first_bit_index)];
      bitmask_type slack_mask = set_least_significant_bits(num_slack_bits);
      delta -= __popc(word & slack_mask);
    }

    // compute delta due to the following bits in the last word in the range

    num_slack_bits = (last_bit_index % word_size_in_bits) == 0
                       ? 0
                       : word_size_in_bits - intra_word_index(last_bit_index);
    if (num_slack_bits > 0) {
      bitmask_type word       = bitmask[word_index(last_bit_index)];
      bitmask_type slack_mask = set_most_significant_bits(num_slack_bits);
      delta -= __popc(word & slack_mask);
    }

    size_type updated_null_count = *(null_counts + range_id) + delta;
    *(null_counts + range_id)    = updated_null_count;

    range_id += blockDim.x * gridDim.x;
  }
}

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
__global__ void copy_offset_bitmask(bitmask_type *__restrict__ destination,
                                    bitmask_type const *__restrict__ source,
                                    size_type source_begin_bit,
                                    size_type source_end_bit,
                                    size_type number_of_mask_words)
{
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    destination[destination_word_index] = detail::get_mask_offset_word(
      source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

/**
 * @brief Computes the bitwise AND of an array of bitmasks
 *
 * @param destination The bitmask to write result into
 * @param source Array of source mask pointers. All masks must be of same size
 * @param begin_bit Array of offsets into corresponding @p source masks.
 *                  Must be same size as source array
 * @param num_sources Number of masks in @p source array
 * @param source_size Number of bits in each mask in @p source
 * @param number_of_mask_words The number of words of type bitmask_type to copy
 */
__global__ void offset_bitmask_and(bitmask_type *__restrict__ destination,
                                   bitmask_type const *const *__restrict__ source,
                                   size_type const *__restrict__ begin_bit,
                                   size_type num_sources,
                                   size_type source_size,
                                   size_type number_of_mask_words)
{
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    bitmask_type destination_word = ~bitmask_type{0};  // All bits 1
    for (size_type i = 0; i < num_sources; i++) {
      destination_word &= detail::get_mask_offset_word(
        source[i], destination_word_index, begin_bit[i], begin_bit[i] + source_size);
    }

    destination[destination_word_index] = destination_word;
  }
}

// convert [first_bit_index,last_bit_index) to
// [first_word_index,last_word_index)
struct to_word_index : public thrust::unary_function<size_type, size_type> {
  const bool _inclusive                 = false;
  size_type const *const _d_bit_indices = nullptr;

  /**
   * @brief Constructor of a functor that converts bit indices to bitmask word
   * indices.
   *
   * @param[in] inclusive Flag that indicates whether bit indices are inclusive
   * or exclusive.
   * @param[in] d_bit_indices Pointer to an array of bit indices
   */
  __host__ to_word_index(bool inclusive, size_type const *d_bit_indices)
    : _inclusive(inclusive), _d_bit_indices(d_bit_indices)
  {
  }

  __device__ size_type operator()(const size_type &i) const
  {
    auto bit_index = _d_bit_indices[i];
    return word_index(bit_index) + ((_inclusive || intra_word_index(bit_index) == 0) ? 0 : 1);
  }
};

}  // namespace

namespace detail {

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const *mask,
                                size_type begin_bit,
                                size_type end_bit,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource *mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit <= end_bit, "Invalid bit range.");
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(end_bit - begin_bit);
  if ((mask == nullptr) || (num_bytes == 0)) { return dest_mask; }
  if (begin_bit == 0) {
    dest_mask = rmm::device_buffer{static_cast<void const *>(mask), num_bytes, stream, mr};
  } else {
    auto number_of_mask_words = num_bitmask_words(end_bit - begin_bit);
    dest_mask                 = rmm::device_buffer{num_bytes, stream, mr};
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      static_cast<bitmask_type *>(dest_mask.data()),
      mask,
      begin_bit,
      end_bit,
      number_of_mask_words);
    CHECK_CUDA(stream.value());
  }
  return dest_mask;
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const &view,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource *mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.nullable()) {
    null_mask =
      copy_bitmask(view.null_mask(), view.offset(), view.offset() + view.size(), stream, mr);
  }
  return null_mask;
}

// Inplace Bitwise AND of the masks
void inplace_bitmask_and(bitmask_type *dest_mask,
                         std::vector<bitmask_type const *> const &masks,
                         std::vector<size_type> const &begin_bits,
                         size_type mask_size,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(std::all_of(begin_bits.begin(), begin_bits.end(), [](auto b) { return b >= 0; }),
               "Invalid range.");
  CUDF_EXPECTS(mask_size > 0, "Invalid bit range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), [](auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");

  auto number_of_mask_words = num_bitmask_words(mask_size);

  rmm::device_vector<bitmask_type const *> d_masks(masks);
  rmm::device_vector<size_type> d_begin_bits(begin_bits);

  cudf::detail::grid_1d config(number_of_mask_words, 256);
  offset_bitmask_and<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    dest_mask,
    d_masks.data().get(),
    d_begin_bits.data().get(),
    d_masks.size(),
    mask_size,
    number_of_mask_words);

  CHECK_CUDA(stream.value());
}

// Bitwise AND of the masks
rmm::device_buffer bitmask_and(std::vector<bitmask_type const *> const &masks,
                               std::vector<size_type> const &begin_bits,
                               size_type mask_size,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource *mr)
{
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(mask_size);

  dest_mask = rmm::device_buffer{num_bytes, stream, mr};
  inplace_bitmask_and(
    static_cast<bitmask_type *>(dest_mask.data()), masks, begin_bits, mask_size, stream, mr);

  return dest_mask;
}

cudf::size_type count_set_bits(bitmask_type const *bitmask,
                               size_type start,
                               size_type stop,
                               rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  if (nullptr == bitmask) { return 0; }

  CUDF_EXPECTS(start >= 0, "Invalid range.");
  CUDF_EXPECTS(start <= stop, "Invalid bit range.");

  std::size_t num_bits_to_count = stop - start;
  if (num_bits_to_count == 0) { return 0; }

  auto num_words = num_bitmask_words(num_bits_to_count);

  constexpr size_type block_size{256};

  cudf::detail::grid_1d grid(num_words, block_size);

  rmm::device_scalar<size_type> non_zero_count(0, stream);

  count_set_bits_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      bitmask, start, stop - 1, non_zero_count.data());

  return non_zero_count.value(stream);
}

cudf::size_type count_unset_bits(bitmask_type const *bitmask,
                                 size_type start,
                                 size_type stop,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  if (nullptr == bitmask) { return 0; }
  auto num_bits = (stop - start);
  return (num_bits - detail::count_set_bits(bitmask, start, stop, stream));
}

std::vector<size_type> segmented_count_set_bits(bitmask_type const *bitmask,
                                                std::vector<size_type> const &indices,
                                                rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(indices.size() % 2 == 0,
               "Array of indices needs to have an even number of elements.");
  for (size_t i = 0; i < indices.size() / 2; i++) {
    auto begin = indices[i * 2];
    auto end   = indices[i * 2 + 1];
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
    CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
  }

  if (indices.empty()) {
    return std::vector<size_type>{};
  } else if (bitmask == nullptr) {
    std::vector<size_type> ret(indices.size() / 2);
    for (size_t i = 0; i < indices.size() / 2; i++) {
      ret[i] = indices[2 * i + 1] - indices[2 * i];
    }
    return ret;
  }

  size_type num_ranges = indices.size() / 2;
  thrust::host_vector<size_type> h_first_indices(num_ranges);
  thrust::host_vector<size_type> h_last_indices(num_ranges);
  thrust::stable_partition_copy(thrust::seq,
                                std::begin(indices),
                                std::end(indices),
                                thrust::make_counting_iterator(0),
                                h_first_indices.begin(),
                                h_last_indices.begin(),
                                [](auto i) { return (i % 2) == 0; });

  rmm::device_vector<size_type> d_first_indices = h_first_indices;
  rmm::device_vector<size_type> d_last_indices  = h_last_indices;
  rmm::device_vector<size_type> d_null_counts(num_ranges, 0);

  auto word_num_set_bits = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [bitmask] __device__(auto i) { return static_cast<size_type>(__popc(bitmask[i])); });
  auto first_word_indices = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    // We cannot use lambda as cub::DeviceSegmentedReduce::Sum() requires
    // first_word_indices and last_word_indices to have the same type.
    to_word_index(true, d_first_indices.data().get()));
  auto last_word_indices = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    // We cannot use lambda as cub::DeviceSegmentedReduce::Sum() requires
    // first_word_indices and last_word_indices to have the same type.
    to_word_index(false, d_last_indices.data().get()));

  // first allocate temporary memroy

  size_t temp_storage_bytes{0};
  CUDA_TRY(cub::DeviceSegmentedReduce::Sum(nullptr,
                                           temp_storage_bytes,
                                           word_num_set_bits,
                                           d_null_counts.begin(),
                                           num_ranges,
                                           first_word_indices,
                                           last_word_indices,
                                           stream.value()));
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  // second perform segmented reduction

  CUDA_TRY(cub::DeviceSegmentedReduce::Sum(d_temp_storage.data(),
                                           temp_storage_bytes,
                                           word_num_set_bits,
                                           d_null_counts.begin(),
                                           num_ranges,
                                           first_word_indices,
                                           last_word_indices,
                                           stream.value()));

  CHECK_CUDA(stream.value());

  // third, adjust counts in segment boundaries (if segments are not
  // word-aligned)

  constexpr size_type block_size{256};

  cudf::detail::grid_1d grid(num_ranges, block_size);

  subtract_set_bits_range_boundaries_kerenel<<<grid.num_blocks,
                                               grid.num_threads_per_block,
                                               0,
                                               stream.value()>>>(
    bitmask, num_ranges, d_first_indices.begin(), d_last_indices.begin(), d_null_counts.begin());

  CHECK_CUDA(stream.value());

  std::vector<size_type> ret(num_ranges);
  CUDA_TRY(cudaMemcpyAsync(ret.data(),
                           d_null_counts.data().get(),
                           num_ranges * sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));

  stream.synchronize();  // now ret is valid.

  return ret;
}

std::vector<size_type> segmented_count_unset_bits(bitmask_type const *bitmask,
                                                  std::vector<size_type> const &indices,
                                                  rmm::cuda_stream_view stream)
{
  if (indices.empty()) {
    return std::vector<size_type>{};
  } else if (bitmask == nullptr) {
    return std::vector<size_type>(indices.size() / 2, 0);
  }

  auto ret = segmented_count_set_bits(bitmask, indices, stream);
  for (size_t i = 0; i < ret.size(); i++) {
    auto begin = indices[i * 2];
    auto end   = indices[i * 2 + 1];
    ret[i]     = (end - begin) - ret[i];
  }

  return ret;
}

// Returns the bitwise AND of the null masks of all columns in the table view
rmm::device_buffer bitmask_and(table_view const &view,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource *mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.num_rows() == 0 or view.num_columns() == 0) { return null_mask; }

  std::vector<bitmask_type const *> masks;
  std::vector<size_type> offsets;
  for (auto &&col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }

  if (masks.size() > 0) {
    return cudf::detail::bitmask_and(masks, offsets, view.num_rows(), stream, mr);
  }

  return null_mask;
}

}  // namespace detail

// Count non-zero bits in the specified range
cudf::size_type count_set_bits(bitmask_type const *bitmask, size_type start, size_type stop)
{
  CUDF_FUNC_RANGE();
  return detail::count_set_bits(bitmask, start, stop);
}

// Count zero bits in the specified range
cudf::size_type count_unset_bits(bitmask_type const *bitmask, size_type start, size_type stop)
{
  CUDF_FUNC_RANGE();
  return detail::count_unset_bits(bitmask, start, stop);
}

// Count non-zero bits in the specified ranges
std::vector<size_type> segmented_count_set_bits(bitmask_type const *bitmask,
                                                std::vector<size_type> const &indices)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_count_set_bits(bitmask, indices, rmm::cuda_stream_default);
}

// Count zero bits in the specified ranges
std::vector<size_type> segmented_count_unset_bits(bitmask_type const *bitmask,
                                                  std::vector<size_type> const &indices)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_count_unset_bits(bitmask, indices, rmm::cuda_stream_default);
}

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const *mask,
                                size_type begin_bit,
                                size_type end_bit,
                                rmm::mr::device_memory_resource *mr)
{
  return detail::copy_bitmask(mask, begin_bit, end_bit, rmm::cuda_stream_default, mr);
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const &view, rmm::mr::device_memory_resource *mr)
{
  return detail::copy_bitmask(view, rmm::cuda_stream_default, mr);
}

rmm::device_buffer bitmask_and(table_view const &view, rmm::mr::device_memory_resource *mr)
{
  return detail::bitmask_and(view, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
