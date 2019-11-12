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

#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
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

template <typename SourceValueIterator, typename SourceValidityIterator,
          typename T, bool has_validity>
__global__
void copy_range_kernel(SourceValueIterator source_value_begin,
                       SourceValidityIterator source_validity_begin,
                       T* __restrict__ const data,
                       cudf::bitmask_type* __restrict__ const bitmask,
                       cudf::size_type* __restrict__ const null_count,
                       cudf::size_type begin,
                       cudf::size_type end) {
  using cudf::experimental::detail::warp_size;

  const cudf::size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr size_t mask_size = warp_size;

  const cudf::size_type masks_per_grid = gridDim.x * blockDim.x / mask_size;
  const int warp_id = tid / warp_size;
  // this assumes that blockDim.x / warp_size <= warp_size
  const int warp_null_change_id = threadIdx.x / warp_size;
  const int lane_id = threadIdx.x % warp_size;

  const cudf::size_type begin_mask_idx = cudf::word_index(begin);
  const cudf::size_type end_mask_idx = cudf::word_index(end);

  cudf::size_type mask_idx = begin_mask_idx + warp_id;

  cudf::size_type target_offset = begin_mask_idx * mask_size - begin;
  cudf::size_type source_idx = tid + target_offset;

  // each warp shares its total change in null count to shared memory to ease
  // computing the total change to null_count.
  // note maximum block size is limited to 1024 by this, but that's OK
  __shared__ uint32_t warp_null_change[has_validity ? warp_size : 1];
  if (has_validity) {
    if (threadIdx.x < warp_size) warp_null_change[threadIdx.x] = 0;
    __syncthreads(); // wait for shared data and validity mask to be complete
  }

  while (mask_idx <= end_mask_idx) {
    cudf::size_type index = mask_idx * mask_size + lane_id;
    bool in_range = (index >= begin && index < end);

    // write data
    if (in_range) data[index] = *(source_value_begin + source_idx);

    if (has_validity) {  // update bitmask
      int active_mask = __ballot_sync(0xFFFFFFFF, in_range);

      bool valid = in_range && *(source_validity_begin + source_idx);
      int warp_mask = __ballot_sync(active_mask, valid);

      cudf::bitmask_type old_mask = bitmask[mask_idx];

      if (lane_id == 0) {
        cudf::bitmask_type new_mask = (old_mask & ~active_mask) |
                                      (warp_mask & active_mask);
        bitmask[mask_idx] = new_mask;
        // null_diff = (mask_size - __popc(new_mask)) - (mask_size - __popc(old_mask))
        warp_null_change[warp_null_change_id] +=
          __popc(active_mask & old_mask) - __popc(active_mask & new_mask);
      }
    }

    source_idx += blockDim.x * gridDim.x;
    mask_idx += masks_per_grid;
  }

  if (has_validity) {
    __syncthreads(); // wait for shared null counts to be ready

    // Compute total null_count change for this block and add it to global count
    if (threadIdx.x < warp_size) {
      uint32_t my_null_change = warp_null_change[threadIdx.x];

      __shared__ typename cub::WarpReduce<uint32_t>::TempStorage temp_storage;

      uint32_t block_null_change =
        cub::WarpReduce<uint32_t>(temp_storage).Sum(my_null_change);

      if (lane_id == 0) { // one thread computes and adds to null count
        atomicAdd(null_count, block_null_change);
      }
    }
  }
}

}  // namespace anonymous

namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Internal API to copy a range of values from a functor to a column.
 *
 * Copies N values from @p source to the range [@p begin, @p end)
 * of @p target. @p target is modified in place.
 *
 * SourceFunctor must have these accessors:
 * __device__ T data(cudf::size_type index);
 * __device__ bool valid(cudf::size_type index);
 *
 * @tparam SourceFunctor the type of the source function object
 * @param source An instance of SourceFunctor that provides data and valid mask
 * @param target the column to copy into
 * @param begin The beginning of the target range to write to
 * @param end The index after the last element of the target range to write to
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

  static_assert(warp_size == cudf::detail::size_in_bits<bitmask_type>(),
    "copy_range_kernel assumes bitmask element size in bits == warp size");

  T * __restrict__ data = target.head<T>();
  bitmask_type * __restrict__ bitmask = target.null_mask();
  auto offset = target.offset();

#if 1
  auto warp_aligned_begin_lower_bound =
    size_type{target_begin - (target_begin % warp_size)};
  auto warp_aligned_end_upper_bound =
    size_type{
      (target_end % warp_size) == 0 ? \
        target_end : target_end + (warp_size - (target_end % warp_size))
    };
  auto num_items =
    warp_aligned_end_upper_bound - warp_aligned_begin_lower_bound;
#else
  // This one results in a compiler internal error! TODO: file NVIDIA bug
  // size_type num_items =
  //   cudf::util::round_up_safe(target_end - target_begin, warp_size);
  // number threads to cover range, rounded to nearest warp
  // this code runs for one additional round if target_begin is not warp
  // aligned, and target_end is block_size + 1
  auto num_items =
    size_type{
      warp_size *
        cudf::util::div_rounding_up_safe(target_end - target_begin, warp_size)
    };
#endif

  constexpr auto block_size = int{256};
  static_assert(block_size <= 1024,
    "copy_range kernel assumes block_size is not larger than 1024");

  auto grid = cudf::experimental::detail::grid_1d{num_items, block_size, 1};

  if (target.nullable()) {
    // TODO: if null_count is UNKNOWN_NULL_COUNT, no need to update null
    // count (if null_count is UNKNOWN_NULL_COUNT, invoking null_count()
    // will scan the entire bitmask array, and this can be surprising
    // in performance if the copy range is small and the column size is
    // large).
    rmm::device_scalar<size_type> null_count(target.null_count(), stream);

    auto kernel =
      copy_range_kernel<SourceValueIterator, SourceValidityIterator, T, true>;
    kernel<<<grid.num_blocks, block_size, 0, stream>>>(
      source_value_begin, source_validity_begin,
      data, bitmask, null_count.data(), offset + target_begin, offset + target_end);

    target.set_null_count(null_count.value());
  }
  else {
    auto kernel =
      copy_range_kernel<SourceValueIterator, SourceValidityIterator, T, false>;
    kernel<<<grid.num_blocks, block_size, 0, stream>>>(
      source_value_begin, source_validity_begin,
      data, bitmask, nullptr, offset + target_begin, offset + target_end);
  }

  CHECK_STREAM(stream);
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
 * @return std::unique_ptr<column> The result target column
 */
std::unique_ptr<column> copy_range(
  column_view const& source,
  column_view const& target,
  size_type source_begin, size_type source_end,
  size_type target_begin,
  cudaStream_t stream = 0,
  rmm::mr::device_memory_resource* mr =
    rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
