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

#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/bit_util.cuh>
#include <utilities/legacy/cuda_utils.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>

#include <cub/cub.cuh>

namespace {

using bit_mask::bit_mask_t;
static constexpr cudf::size_type warp_size{32};

template <typename T, typename InputFunctor, bool has_validity>
__global__
void copy_range_kernel(T * __restrict__ const data,
                       bit_mask_t * __restrict__ const bitmask,
                       cudf::size_type * __restrict__ const null_count,
                       cudf::size_type begin,
                       cudf::size_type end,
                       InputFunctor input)
{
  const cudf::size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr size_t mask_size = warp_size;

  const cudf::size_type masks_per_grid = gridDim.x * blockDim.x / mask_size;
  const int warp_id = tid / warp_size;
  const int warp_null_change_id = threadIdx.x / warp_size;
  const int lane_id = threadIdx.x % warp_size;

  const cudf::size_type begin_mask_idx =
      cudf::util::detail::bit_container_index<bit_mask_t>(begin);
  const cudf::size_type end_mask_idx =
      cudf::util::detail::bit_container_index<bit_mask_t>(end);

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

  while (mask_idx <= end_mask_idx)
  {
    cudf::size_type index = mask_idx * mask_size + lane_id;
    bool in_range = (index >= begin && index < end);

    // write data
    if (in_range) data[index] = input.data(input_idx);

    if (has_validity) { // update bitmask
      int active_mask = __ballot_sync(0xFFFFFFFF, in_range);

      bool valid = in_range and input.valid(input_idx);
      int warp_mask = __ballot_sync(active_mask, valid);

      bit_mask_t old_mask = bitmask[mask_idx];

      if (lane_id == 0) {
        bit_mask_t new_mask = (old_mask & ~active_mask) |
                              (warp_mask & active_mask);
        bitmask[mask_idx] = new_mask;
        // null_diff = (mask_size - __popc(new_mask)) - (mask_size - __popc(old_mask))
        warp_null_change[warp_null_change_id] += __popc(active_mask & old_mask) -
                                                 __popc(active_mask & new_mask);
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
  void operator()(gdf_column *column,
                  cudf::size_type begin, cudf::size_type end,
                  cudaStream_t stream = 0)
  {
    static_assert(warp_size == cudf::util::size_in_bits<bit_mask_t>(), 
      "copy_range_kernel assumes bitmask element size in bits == warp size");

    auto input = factory.template make<T>();
    auto kernel = copy_range_kernel<T, decltype(input), false>;

    cudf::size_type *null_count = nullptr;

    if (cudf::is_nullable(*column)) {
      RMM_ALLOC(&null_count, sizeof(cudf::size_type), stream);
      CUDA_TRY(cudaMemcpyAsync(null_count, &column->null_count, 
                               sizeof(cudf::size_type), 
                               cudaMemcpyHostToDevice,
                               stream));
      kernel = copy_range_kernel<T, decltype(input), true>;
    }

    // This one results in a compiler internal error! TODO: file NVIDIA bug
    // cudf::size_type num_items = cudf::util::round_up_safe(end - begin, warp_size);
    // number threads to cover range, rounded to nearest warp
    cudf::size_type num_items =
      warp_size * cudf::util::div_rounding_up_safe(end - begin, warp_size);

    constexpr int block_size = 256;

    cudf::util::cuda::grid_config_1d grid{num_items, block_size, 1};

    T * __restrict__ data = static_cast<T*>(column->data);
    bit_mask_t * __restrict__ bitmask =
      reinterpret_cast<bit_mask_t*>(column->valid);
  
    kernel<<<grid.num_blocks, block_size, 0, stream>>>
      (data, bitmask, null_count, begin, end, input);

    if (null_count != nullptr) {
      CUDA_TRY(cudaMemcpyAsync(&column->null_count, null_count,
                               sizeof(cudf::size_type), cudaMemcpyDefault, stream));
      RMM_FREE(null_count, stream);
    }

    CHECK_CUDA(stream);
  }
};

}; // namespace anonymous

namespace cudf {

namespace detail {

/**
 * @brief Copies a range of values from a functor to a column
 * 
 * Copies N values from @p input to the range [@p begin, @p end)
 * of @p out_column. @p out_column is modified in place.
 * 
 * InputFunctor must have these accessors:
 * __device__ T data(cudf::size_type index);
 * __device__ bool valid(cudf::size_type index);
 * 
 * @tparam InputFunctor the type of the input function object
 * @p out_column the column to copy into
 * @p input An instance of InputFunctor that provides data and valid mask
 * @p begin The beginning of the output range to write to
 * @p end The index after the last element of the output range to write to
 */
template <typename InputFunctor>
void copy_range(gdf_column *out_column, InputFunctor input,
                cudf::size_type begin, cudf::size_type end)
{
  validate(out_column);
  CUDF_EXPECTS(end - begin > 0, "Range is empty or reversed");
  CUDF_EXPECTS((begin >= 0) and (end <= out_column->size), "Range is out of bounds");

  cudf::type_dispatcher(out_column->dtype,
                        copy_range_dispatch<InputFunctor>{input},
                        out_column, begin, end);

  // synchronize nvcategory after filtering
  if (out_column->dtype == GDF_STRING_CATEGORY) {
    CUDF_EXPECTS(
    GDF_SUCCESS ==
      nvcategory_gather(out_column,
                        static_cast<NVCategory *>(out_column->dtype_info.category)),
      "could not set nvcategory");
  }
}

}; // namespace detail

}; // namespace cudf
