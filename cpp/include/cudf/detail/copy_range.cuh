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

#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <memory>

namespace {

// TODO: better define warp_size elsewhere (cuda_utils.hpp?).
static constexpr cudf::size_type warp_size{32};

template <typename T, typename InputFunctor, bool has_validity>
__global__
void copy_range_kernel(T* __restrict__ const data,
                       cudf::bitmask_type* __restrict__ const bitmask,
                       cudf::size_type* __restrict__ const null_count,
                       cudf::size_type begin,
                       cudf::size_type end,
                       InputFunctor input) {
  const cudf::size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr size_t mask_size = warp_size;

  const cudf::size_type masks_per_grid = gridDim.x * blockDim.x / mask_size;
  const int warp_id = tid / warp_size;
  // this assumes that blockDim.x / warp_size <= warp_size
  const int warp_null_change_id = threadIdx.x / warp_size;
  const int lane_id = threadIdx.x % warp_size;

  const cudf::size_type begin_mask_idx =
      cudf::util::detail::bit_container_index<cudf::bitmask_type>(begin);
  const cudf::size_type end_mask_idx =
      cudf::util::detail::bit_container_index<cudf::bitmask_type>(end);

  cudf::size_type mask_idx = begin_mask_idx + warp_id;

  cudf::size_type output_offset = begin_mask_idx * mask_size - begin;
  cudf::size_type input_idx = tid + output_offset;

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
    if (in_range) data[index] = input.data(input_idx);

    if (has_validity) {  // update bitmask
      int active_mask = __ballot_sync(0xFFFFFFFF, in_range);

      bool valid = in_range and input.valid(input_idx);
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

    input_idx += blockDim.x * gridDim.x;
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

template <typename InputFactory>
struct copy_range_dispatch {
  InputFactory factory;

  template <typename T>
  void operator()(cudf::mutable_column_view& output,
                  cudf::size_type begin, cudf::size_type end,
                  cudaStream_t stream = 0) {
    static_assert(warp_size == cudf::util::size_in_bits<cudf::bitmask_type>(),
      "copy_range_kernel assumes bitmask element size in bits == warp size");

    auto input = factory.template make<T>();

    T * __restrict__ data = output.head<T>();
    cudf::bitmask_type * __restrict__ bitmask = output.null_mask();
    auto offset = output.offset();

#if 1
    auto warp_aligned_begin_lower_bound =
      cudf::size_type{begin - (begin % warp_size)};
    auto warp_aligned_end_upper_bound =
      cudf::size_type{
        (end % warp_size) == 0 ? end : end + (warp_size - (end % warp_size))};
    auto num_items =
      warp_aligned_end_upper_bound - warp_aligned_begin_lower_bound;
#else
    // This one results in a compiler internal error! TODO: file NVIDIA bug
    // cudf::size_type num_items = cudf::util::round_up_safe(end - begin, warp_size);
    // number threads to cover range, rounded to nearest warp
    // this code runs for one additional round if begin is not warp aligned,
    // and end is block_size + 1
    auto num_items =
      cudf::size_type{
        warp_size * cudf::util::div_rounding_up_safe(end - begin, warp_size)};
#endif

    constexpr auto block_size = int{256};
    static_assert(block_size <= 1024,
      "copy_range kernel assumes block_size is not larger than 1024");

    auto grid = cudf::util::cuda::grid_config_1d{num_items, block_size, 1};

    if (output.nullable() == true) {
      // TODO: if null_count is UNKNOWN_NULL_COUNT, no need to update null
      // count (if null_count is UNKNOWN_NULL_COUNT, invoking null_count()
      // will scan the entire bitmask array, and this can be surprising
      // in performance if the copy range is small and the column size is
      // large).
      rmm::device_scalar<cudf::size_type> null_count(output.null_count(), stream);

      auto kernel = copy_range_kernel<T, decltype(input), true>;
      kernel<<<grid.num_blocks, block_size, 0, stream>>>
        (data, bitmask, null_count.data(), offset + begin, offset + end, input);

      output.set_null_count(null_count.value());
    }
    else {
      auto kernel = copy_range_kernel<T, decltype(input), false>;
      kernel<<<grid.num_blocks, block_size, 0, stream>>>
        (data, bitmask, nullptr, offset + begin, offset + end, input);
    }

    CHECK_STREAM(stream);
  }
};

}  // namespace anonymous

namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Internal API to copy a range of values from a functor to a column.
 *
 * Copies N values from @p input to the range [@p begin, @p end)
 * of @p output. @p output is modified in place.
 *
 * InputFunctor must have these accessors:
 * __device__ T data(cudf::size_type index);
 * __device__ bool valid(cudf::size_type index);
 *
 * @tparam InputFunctor the type of the input function object
 * @param output the column to copy into
 * @param input An instance of InputFunctor that provides data and valid mask
 * @param begin The beginning of the output range to write to
 * @param end The index after the last element of the output range to write to
 * @param stream CUDA stream to run this function
 */
template <typename InputFunctor>
void copy_range(mutable_column_view& output, InputFunctor input,
                size_type begin, size_type end, cudaStream_t stream = 0) {
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < output.size()) &&
               (end <= output.size()),
               "Range is out of bounds.");

  type_dispatcher(output.type(),
                  copy_range_dispatch<InputFunctor>{input},
                  output, begin, end, stream);
}

/**
 * @brief Internal API to copy a range of elements in-place from one column to
 * another.
 *
 * Copies N elements of @p input starting at @p in_begin to the N
 * elements of @p output starting at @p out_begin, where
 * N = (@p out_end - @p out_begin).
 *
 * Overwrites the range of elements in @p output indicated by the indices
 * [@p out_begin, @p out_ned) with the elements from @p input indicated by the
 * indices [@p in_begin, @p in_begin + N) (where N =
 * (@p out_end - @p out_begin)). Use the out-of-place copy function returning
 * std::unique_ptr<column> for uses cases requiring memory reallocation.
 *
 * If @p input and @p output refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws `cudf::logic_error` if memory reallocation is required (e.g. for
 * variable width types).
 * @throws `cudf::logic_error` for invalid range (if @p out_begin < 0,
 * @p out_begin > @p out_end, @p out_begin >= @p output.size(),
 * @p out_end > @p output.size(), @p in_begin < 0, in_begin >= @p input.size(),
 * or @p in_begin + @p out_end - @p out_begin > @p input.size()).
 * @throws `cudf::logic_error` if @p output and @p input have different types.
 * @throws `cudf::logic_error` if @p input has null values and @p output is not
 * nullable.
 *
 * @param output The preallocated column to copy into
 * @param input The column to copy from
 * @param out_begin The starting index of the output range (inclusive)
 * @param out_end The index of the last element in the output range (exclusive)
 * @param in_begin The starting index of the input range (inclusive)
 * @param stream CUDA stream to run this function
 * @return void
 */
void copy_range(mutable_column_view& output, column_view const& input,
                size_type out_begin, size_type out_end, size_type in_begin,
                cudaStream_t stream = 0);

/**
 * @brief Internal API to copy a range of elements out-of-place from one column
 * to another.
 *
 * Creates a new column as-if an in-place copy was performed into @p output.
 * A copy of @p output is created first and then the elements indicated by the
 * indices [@p out_begin, @p out_end) were copied from the elements indicated
 * by the indices [@p in_begin, @p in_begin +N) of @p input (where N =
 * (@p out_end - @p out_begin)). Elements outside the range are copied from
 * @p output into the returned new column output.
 *
 * If @p input and @p output refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws `cudf::logic_error` for invalid range (if @p out_begin < 0,
 * @p out_begin > @p out_end, @p out_begin >= @p output.size(),
 * @p out_end > @p output.size(), @p in_begin < 0, in_begin >= @p input.size(),
 * or @p in_begin + @p out_end - @p out_begin > @p input.size()).
 * @throws `cudf::logic_error` if @p output and @p input have different types.
 *
 * @param output The column to copy from outside the range.
 * @param input The column to copy from inside the range.
 * @param out_begin The starting index of the output range (inclusive)
 * @param out_end The index of the last element in the output range (exclusive)
 * @param in_begin The starting index of the input range (inclusive)
 * @param mr Memory resource to allocate the result output column.
 * @return std::unique_ptr<column> The result output column
 */
std::unique_ptr<column> copy_range(
  column_view const& output,
  column_view const& input,
  size_type out_begin, size_type out_end,
  size_type in_begin,
  cudaStream_t stream = 0,
  rmm::mr::device_memory_resource* mr =
    rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
