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

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <memory>

namespace {

template <cudf::size_type block_size,
          typename SourceValueIterator, typename SourceValidityIterator,
          typename T, bool has_validity>
__global__
void copy_range_kernel(SourceValueIterator source_value_begin,
                       SourceValidityIterator source_validity_begin,
                       cudf::mutable_column_device_view target,
                       cudf::size_type target_begin,
                       cudf::size_type target_end,
                       cudf::size_type* __restrict__ const null_count) {
  using cudf::experimental::detail::warp_size;

  static_assert(block_size <= 1024,
    "copy_range_kernel assumes block_size is not larger than 1024");
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
    "copy_range_kernel assumes bitmask element size in bits == warp size");

  constexpr cudf::size_type leader_lane{0};
  const int lane_id = threadIdx.x % warp_size;

  const cudf::size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int warp_id = tid / warp_size;

  const cudf::size_type offset = target.offset();
  const cudf::size_type begin_mask_idx =
    cudf::word_index(offset + target_begin);
  const cudf::size_type end_mask_idx = cudf::word_index(offset + target_end);

  cudf::size_type mask_idx = begin_mask_idx + warp_id;
  const cudf::size_type masks_per_grid = gridDim.x * blockDim.x / warp_size;

  cudf::size_type target_offset =
    begin_mask_idx * warp_size - (offset + target_begin);
  cudf::size_type source_idx = tid + target_offset;

  cudf::size_type warp_null_change{0};

  while (mask_idx <= end_mask_idx) {
    cudf::size_type index = mask_idx * warp_size + lane_id - offset;
    bool in_range = (index >= target_begin && index < target_end);

    // write data
    if (in_range) target.element<T>(index) = *(source_value_begin + source_idx);

    if (has_validity) {  // update bitmask
      int active_mask = __ballot_sync(0xFFFFFFFF, in_range);

      bool valid = in_range && *(source_validity_begin + source_idx);
      int warp_mask = __ballot_sync(active_mask, valid);

      cudf::bitmask_type old_mask = target.get_mask_word(mask_idx);

      if (lane_id == leader_lane) {
        cudf::bitmask_type new_mask = (old_mask & ~active_mask) |
                                      (warp_mask & active_mask);
        target.set_mask_word(mask_idx, new_mask);
        // null_diff =
        //   (warp_size - __popc(new_mask)) - (warp_size - __popc(old_mask))
        warp_null_change +=
          __popc(active_mask & old_mask) - __popc(active_mask & new_mask);
      }
    }

    source_idx += blockDim.x * gridDim.x;
    mask_idx += masks_per_grid;
  }

  if (has_validity) {
    auto block_null_change =
      cudf::experimental::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_null_change);
    if (threadIdx.x == 0) {  // if the first thread in a block
      atomicAdd(null_count, block_null_change);
    }
  }
}

}  // namespace anonymous

namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Internal API to copy a range of values from source iterators to a
 * target column.
 *
 * The elements indicated by the indices [@p target_begin, @p target_end) were
 * replaced with the elements retrieved from source iterators;
 * *(@p source_value_begin + idx) if *(@p source_validity_begin + idx) is true,
 * invalidate otherwise (where idx = [0, @p target_end - @p target_begin)).
 * @p target is modified in place.
 *
 * @tparam SourceValueIterator Iterator for retrieving source values
 * @tparam SourceValidityIterator Iterator for retrieving source validities
 * @param source_value_begin Start of source value iterator
 * @param source_validity_begin Start of source validity iterator
 * @param target the column to copy into
 * @param target_begin The starting index of the target range (inclusive)
 * @param target_end The index of the last element in the target range
 * (exclusive)
 * @param stream CUDA stream to run this function
 */
template <typename SourceValueIterator, typename SourceValidityIterator>
void copy_range(SourceValueIterator source_value_begin,
                SourceValidityIterator source_validity_begin,
                mutable_column_view& target,
                size_type target_begin, size_type target_end,
                cudaStream_t stream = 0) {
  CUDF_EXPECTS((target_begin <= target_end) &&
                 (target_begin >= 0) &&
                 (target_begin < target.size()) &&
                 (target_end <= target.size()),
               "Range is out of bounds.");
  using T = typename std::iterator_traits<SourceValueIterator>::value_type;

  // this code assumes that source and target have the same type.
  CUDF_EXPECTS(type_to_id<T>() == target.type().id(), "the data type mismatch");

  auto warp_aligned_begin_lower_bound =
    cudf::util::round_down_safe(target_begin, warp_size);
  auto warp_aligned_end_upper_bound =
    cudf::util::round_up_safe(target_end, warp_size);
  auto num_items =
    warp_aligned_end_upper_bound - warp_aligned_begin_lower_bound;

  constexpr size_type block_size{256};

  auto grid = cudf::experimental::detail::grid_1d{num_items, block_size, 1};

  if (target.nullable()) {
    // TODO: if null_count is UNKNOWN_NULL_COUNT, no need to update null
    // count (if null_count is UNKNOWN_NULL_COUNT, invoking null_count()
    // will scan the entire bitmask array, and this can be surprising
    // in performance if the copy range is small and the column size is
    // large).
    rmm::device_scalar<size_type> null_count(target.null_count(), stream);

    auto kernel =
      copy_range_kernel<block_size, SourceValueIterator, SourceValidityIterator,
                        T, true>;
    kernel<<<grid.num_blocks, block_size, 0, stream>>>(
      source_value_begin, source_validity_begin,
      *mutable_column_device_view::create(target, stream),
      target_begin, target_end, null_count.data());

    target.set_null_count(null_count.value());
  }
  else {
    auto kernel =
      copy_range_kernel<block_size, SourceValueIterator, SourceValidityIterator,
                        T, false>;
    kernel<<<grid.num_blocks, block_size, 0, stream>>>(
      source_value_begin, source_validity_begin,
      *mutable_column_device_view::create(target, stream),
      target_begin, target_end, nullptr);
  }

  CHECK_CUDA(stream);
}

/**
 * @brief Internal API to copy a range of elements in-place from one column to
 * another.
 *
 * Overwrites the range of elements in @p target indicated by the indices
 * [@p target_begin, @p target_begin + N) with the elements from @p source
 * indicated by the indices [@p source_begin, @p source_end) (where N =
 * (@p source_end - @p source_begin)). Use the out-of-place copy function
 * returning std::unique_ptr<column> for uses cases requiring memory
 * reallocation. For example for strings columns and other variable-width types.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws `cudf::logic_error` if memory reallocation is required (e.g. for
 * variable width types).
 * @throws `cudf::logic_error` for invalid range (if
 * @p source_begin > @p source_end, @p source_begin < 0,
 * @p source_begin >= @p source.size(), @p source_end > @p source.size(),
 * @p target_begin < 0, target_begin >= @p target.size(), or
 * @p target_begin + (@p source_end - @p source_begin) > @p target.size()).
 * @throws `cudf::logic_error` if @p target and @p source have different types.
 * @throws `cudf::logic_error` if @p source has null values and @p target is not
 * nullable.
 *
 * @param source The column to copy from
 * @param target The preallocated column to copy into
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @return void
 */
void copy_range(column_view const& source, mutable_column_view& target,
                size_type source_begin, size_type source_end,
                size_type target_begin,
                cudaStream_t stream = 0);

/**
 * @brief Internal API to copy a range of elements out-of-place from one column
 * to another.
 *
 * Creates a new column as if an in-place copy was performed into @p target.
 * A copy of @p target is created first and then the elements indicated by the
 * indices [@p target_begin, @p target_begin + N) were copied from the elements
 * indicated by the indices [@p source_begin, @p source_end) of @p source
 * (where N = (@p source_end - @p source_begin)). Elements outside the range are
 * copied from @p target into the returned new column target.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws `cudf::logic_error` for invalid range (if
 * @p source_begin > @p source_end, @p source_begin < 0,
 * @p source_begin >= @p source.size(), @p source_end > @p source.size(),
 * @p target_begin < 0, target_begin >= @p target.size(), or
 * @p target_begin + (@p source_end - @p source_begin) > @p target.size()).
 * @throws `cudf::logic_error` if @p target and @p source have different types.
 *
 * @param source The column to copy from inside the range.
 * @param target The column to copy from outside the range.
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param mr Memory resource to allocate the result target column.
 * @param stream CUDA stream to run this function
 * @return std::unique_ptr<column> The result target column
 */
std::unique_ptr<column> copy_range(
  column_view const& source,
  column_view const& target,
  size_type source_begin, size_type source_end,
  size_type target_begin,
  rmm::mr::device_memory_resource* mr =
    rmm::mr::get_default_resource(),
  cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
