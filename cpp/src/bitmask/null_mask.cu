/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/null_mask.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/cuda.cuh>


#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <cub/cub.cuh>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace cudf {

size_type state_null_count(mask_state state, size_type size) {
  switch (state) {
    case UNALLOCATED:
      return 0;
    case UNINITIALIZED:
      return UNKNOWN_NULL_COUNT;
    case ALL_NULL:
      return size;
    case ALL_VALID:
      return 0;
    default:
      CUDF_FAIL("Invalid null mask state.");
  }
}

// Computes required allocation size of a bitmask
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                          std::size_t padding_boundary) {
  CUDF_EXPECTS(padding_boundary > 0, "Invalid padding boundary");
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}

// Computes number of *actual* bitmask_type elements needed
size_type num_bitmask_words(size_type number_of_bits) {
  return cudf::util::div_rounding_up_safe<size_type>(
      number_of_bits, detail::size_in_bits<bitmask_type>());
}

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size, mask_state state,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource *mr) {
  size_type mask_size{0};

  if (state != UNALLOCATED) {
    mask_size = bitmask_allocation_size_bytes(size);
  }

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != UNINITIALIZED) {
    uint8_t fill_value = (state == ALL_VALID) ? 0xff : 0x00;
    CUDA_TRY(cudaMemsetAsync(static_cast<bitmask_type *>(mask.data()),
                             fill_value, mask_size, stream));
  }

  return mask;
}

//Set pre-allocated null mask to:
//all entries to valid, if valid_flag==true,
//or null, otherwise;
void set_null_mask(bitmask_type* bitmask,
                   size_type size, bool valid,
                   cudaStream_t stream)
{
  if (bitmask != nullptr) {
    size_type mask_size = bitmask_allocation_size_bytes(size);

    uint8_t fill_value = (valid == true) ? 0xff : 0x00;
    CUDA_TRY(cudaMemsetAsync(bitmask,
                             fill_value, mask_size, stream));
  }
}

namespace {

/**---------------------------------------------------------------------------*
 * @brief Counts the number of non-zero bits in a bitmask in the range
 * `[first_bit_index, last_bit_index]`.
 *
 * Expects `0 <= first_bit_index <= last_bit_index`.
 *
 * @param[in] bitmask The bitmask whose non-zero bits will be counted.
 * @param[in] first_bit_index The index (inclusive) of the first bit to count
 * @param[in] last_bit_index The index (inclusive) of the last bit to count
 * @param[out] global_count The number of non-zero bits in the specified range
 *---------------------------------------------------------------------------**/
template <size_type block_size>
__global__ void count_set_bits_kernel(bitmask_type const *bitmask,
                                      size_type first_bit_index,
                                      size_type last_bit_index,
                                      size_type *global_count) {
  constexpr auto const word_size{detail::size_in_bits<bitmask_type>()};

  auto const first_word_index{word_index(first_bit_index)};
  auto const last_word_index{word_index(last_bit_index)};
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
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

    size_type bit_index = (first) ? first_bit_index : last_bit_index;
    size_type word_index = (first) ? first_word_index : last_word_index;

    size_type num_slack_bits = bit_index % word_size;
    if (last) {
      num_slack_bits = word_size - num_slack_bits - 1;
    }

    if (num_slack_bits > 0) {
      bitmask_type word = bitmask[word_index];
      auto slack_mask = (first) ? set_least_significant_bits(num_slack_bits)
                                : set_most_significant_bits(num_slack_bits);

      thread_count -= __popc(word & slack_mask);
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_count{BlockReduce(temp_storage).Sum(thread_count)};

  if (threadIdx.x == 0) {
    atomicAdd(global_count, block_count);
  }
}

/**
 * @brief Convenience function to get offset word from a bitmask
 * 
 * @see copy_offset_bitmask
 * @see offset_bitmask_and
 */
__device__ bitmask_type get_mask_offset_word(
  bitmask_type const *__restrict__ source,
  size_type destination_word_index,
  size_type source_begin_bit,
  size_type source_end_bit)
{
  size_type source_word_index = destination_word_index + word_index(source_begin_bit);
  bitmask_type curr_word = source[source_word_index];
  bitmask_type next_word = 0;
  if ((word_index(source_begin_bit) != 0) &&
      (word_index(source_end_bit) >
        word_index(source_begin_bit +
          destination_word_index * detail::size_in_bits<bitmask_type>()))) {
    next_word = source[source_word_index + 1];
  }
  return __funnelshift_r(curr_word, next_word, source_begin_bit);
}

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
// TODO: Also make binops test that uses offset in column_view
__global__ void copy_offset_bitmask(bitmask_type *__restrict__ destination,
                                    bitmask_type const *__restrict__ source,
                                    size_type source_begin_bit,
                                    size_type source_end_bit,
                                    size_type number_of_mask_words) {
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    destination[destination_word_index] =
      get_mask_offset_word(source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @param views Array of column_device_view
 * @param output_offsets Prefix sum of sizes of elements of `views`
 * @param number_of_views Size of `views` array
 * @param dest_mask The output buffer to copy null masks into
 * @param number_of_mask_bits The total number of null masks bits that are being
 * copied
 *---------------------------------------------------------------------------**/
__global__
void
concatenate_masks_kernel(
    column_device_view* views,
    size_type* output_offsets,
    size_type number_of_views,
    bitmask_type* dest_mask,
    size_type number_of_mask_bits) {

  size_type mask_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask =
      __ballot_sync(0xFFFF'FFFF, mask_index < number_of_mask_bits);

  while (mask_index < number_of_mask_bits) {
    size_type source_view_index = thrust::upper_bound(thrust::seq,
        output_offsets, output_offsets + number_of_views,
        mask_index) - output_offsets - 1;
    bool bit_is_set = 1;
    if (source_view_index < number_of_views) {
      size_type column_element_index = mask_index - output_offsets[source_view_index];
      bit_is_set = views[source_view_index].is_valid(column_element_index);
    }
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    if (threadIdx.x % experimental::detail::warp_size == 0) {
      dest_mask[word_index(mask_index)] = new_word;
    }

    mask_index += blockDim.x * gridDim.x;
    active_mask =
        __ballot_sync(active_mask, mask_index < number_of_mask_bits);
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
                                   bitmask_type const * const*__restrict__ source,
                                   size_type const* __restrict__ begin_bit,
                                   size_type num_sources, 
                                   size_type source_size,
                                   size_type number_of_mask_words) {
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {

    bitmask_type destination_word = ~bitmask_type{0}; // All bits 1
    for (size_type i = 0; i < num_sources; i++) {
      destination_word &= get_mask_offset_word(
        source[i], destination_word_index, begin_bit[i], begin_bit[i] + source_size);
    }

    destination[destination_word_index] = destination_word;
  }
}

// Bitwise AND of the masks
rmm::device_buffer bitmask_and(std::vector<bitmask_type const*> const& masks, 
                               std::vector<size_type> const& begin_bits,
                               size_type mask_size,
                               cudaStream_t stream,
                               rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(std::all_of(begin_bits.begin(), begin_bits.end(), 
                           [] (auto b) { return b >= 0; }),
               "Invalid range.");
  CUDF_EXPECTS(mask_size > 0, "Invalid bit range.");
  CUDF_EXPECTS(std::all_of(masks.begin(), masks.end(), 
                           [] (auto p) { return p != nullptr; }),
               "Mask pointer cannot be null");

  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(mask_size);

  auto number_of_mask_words = num_bitmask_words(mask_size);

  dest_mask = rmm::device_buffer{num_bytes, stream, mr};

  rmm::device_vector<bitmask_type const *> d_masks(masks);
  rmm::device_vector<size_type> d_begin_bits(begin_bits);
  
  cudf::experimental::detail::grid_1d config(number_of_mask_words, 256);
  offset_bitmask_and<<<config.num_blocks, config.num_threads_per_block, 0,
                        stream>>>(
      static_cast<bitmask_type *>(dest_mask.data()), 
      d_masks.data().get(), d_begin_bits.data().get(),
      d_masks.size(), mask_size, number_of_mask_words);
  
  CHECK_CUDA(stream);

  return dest_mask;
}

}  // namespace

namespace detail {
cudf::size_type count_set_bits(bitmask_type const *bitmask, size_type start,
                               size_type stop, cudaStream_t stream = 0) {
  if (nullptr == bitmask) {
    return 0;
  }

  CUDF_EXPECTS(start >= 0, "Invalid range.");
  CUDF_EXPECTS(start <= stop, "Invalid bit range.");

  std::size_t num_bits_to_count = stop - start;
  if (num_bits_to_count == 0) {
    return 0;
  }

  auto num_words = num_bitmask_words(num_bits_to_count);

  constexpr size_type block_size{256};

  cudf::experimental::detail::grid_1d grid(num_words, block_size);

  rmm::device_scalar<size_type> non_zero_count(0, stream);

  count_set_bits_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
          bitmask, start, stop - 1, non_zero_count.data());

  return non_zero_count.value();
}

cudf::size_type count_unset_bits(bitmask_type const *bitmask, size_type start,
                                 size_type stop, cudaStream_t stream = 0) {
  if (nullptr == bitmask) {
    return 0;
  }
  auto num_bits = (stop - start);
  return (num_bits - detail::count_set_bits(bitmask, start, stop, stream));
}

// Create a bitmask from a vector of column views
void concatenate_masks(std::vector<column_view> const &views,
    bitmask_type * dest_mask,
    cudaStream_t stream) {
  using CDViewPtr =
    decltype(column_device_view::create(std::declval<column_view>(), std::declval<cudaStream_t>()));
  std::vector<CDViewPtr> cols;
  thrust::host_vector<column_device_view> device_views;

  thrust::host_vector<size_type> view_offsets(1, 0);
  for (auto &v : views) {
    cols.emplace_back(column_device_view::create(v, stream));
    device_views.push_back(*(cols.back()));
    view_offsets.push_back(v.size());
  }
  thrust::inclusive_scan(thrust::host,
      view_offsets.begin(), view_offsets.end(),
      view_offsets.begin());

  rmm::device_vector<column_device_view> d_views{device_views};
  rmm::device_vector<size_type> d_offsets{view_offsets};

  auto number_of_mask_bits = view_offsets.back();
  constexpr size_type block_size{256};
  cudf::experimental::detail::grid_1d config(number_of_mask_bits, block_size);
  concatenate_masks_kernel<<<config.num_blocks, config.num_threads_per_block,
                             0, stream>>>(
    d_views.data().get(),
    d_offsets.data().get(),
    static_cast<size_type>(d_views.size()),
    dest_mask, number_of_mask_bits);
}

}  // namespace detail

// Count non-zero bits in the specified range
cudf::size_type count_set_bits(bitmask_type const *bitmask, size_type start,
                               size_type stop) {
  return detail::count_set_bits(bitmask, start, stop);
}

// Count zero bits in the specified range
cudf::size_type count_unset_bits(bitmask_type const *bitmask, size_type start,
                                 size_type stop) {
  return detail::count_unset_bits(bitmask, start, stop);
}

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const *mask, size_type begin_bit,
                                size_type end_bit, cudaStream_t stream,
                                rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit <= end_bit, "Invalid bit range.");
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(end_bit - begin_bit);
  if ((mask == nullptr) || (num_bytes == 0)) {
    return dest_mask;
  }
  if (begin_bit == 0) {
    dest_mask = rmm::device_buffer{static_cast<void const *>(mask), num_bytes,
                                   stream, mr};
  } else {
    auto number_of_mask_words = cudf::util::div_rounding_up_safe(
        static_cast<size_t>(end_bit - begin_bit),
        detail::size_in_bits<bitmask_type>());
    dest_mask = rmm::device_buffer{num_bytes, stream, mr};
    cudf::experimental::detail::grid_1d config(number_of_mask_words, 256);
    copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0,
                          stream>>>(
        static_cast<bitmask_type *>(dest_mask.data()), mask, begin_bit, end_bit,
        number_of_mask_words);
    CHECK_CUDA(stream);
  }
  return dest_mask;
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const &view, cudaStream_t stream,
                                rmm::mr::device_memory_resource *mr) {
  rmm::device_buffer null_mask{};
  if (view.nullable()) {
    null_mask = copy_bitmask(view.null_mask(), view.offset(),
                             view.offset() + view.size(), stream, mr);
  }
  return null_mask;
}

// Create a bitmask from a vector of column views
rmm::device_buffer concatenate_masks(std::vector<column_view> const &views,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {
  rmm::device_buffer null_mask{};
  bool has_nulls = std::any_of(views.begin(), views.end(),
                     [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
   size_type total_element_count =
     std::accumulate(views.begin(), views.end(), 0,
         [](auto accumulator, auto const& v) { return accumulator + v.size(); });
    null_mask = create_null_mask(total_element_count, UNINITIALIZED, stream, mr);

    detail::concatenate_masks(
        views, static_cast<bitmask_type *>(null_mask.data()), stream);
  }

  return null_mask;
}

// Returns the bitwise AND of the null masks of all columns in the table view
rmm::device_buffer bitmask_and(table_view const& view,
                               rmm::mr::device_memory_resource *mr,
                               cudaStream_t stream) {
  rmm::device_buffer null_mask{};
  if (view.num_rows() == 0 or view.num_columns() == 0) {
    return null_mask;
  }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> offsets;
  for (auto &&col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }
  
  if (masks.size() > 0) {
    return bitmask_and(masks, offsets, view.num_rows(), stream, mr);
  }
  
  return null_mask;
}

}  // namespace cudf
