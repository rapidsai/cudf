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

#include <filling.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>

namespace {

using bit_mask::bit_mask_t;
static constexpr gdf_size_type warp_size{32};

template <typename T, bool has_nulls>
__global__ 
void fill_kernel(T *data, bit_mask_t *bitmask, 
                 gdf_size_type *null_count,
                 gdf_index_type begin, gdf_index_type end,
                 T value, bool value_is_valid)
{
  const gdf_index_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr size_t mask_size = warp_size;

  const gdf_size_type masks_per_grid = gridDim.x * blockDim.x / mask_size;
  const int warp_id = tid / warp_size;
  const int lane_id = threadIdx.x % warp_size;

  const gdf_index_type begin_mask_idx =
      cudf::util::detail::bit_container_index<bit_mask_t>(begin);
  const gdf_index_type end_mask_idx =
      cudf::util::detail::bit_container_index<bit_mask_t>(end);

  gdf_index_type mask_idx = begin_mask_idx + warp_id;

  while (mask_idx <= end_mask_idx)
  {
    gdf_index_type index = mask_idx * mask_size + lane_id;
    bool in_range = (index >= begin && index < end);

    // write data
    if (in_range) data[index] = value;

    if (has_nulls) { // update bitmask
      int active_mask = __ballot_sync(0xFFFFFFFF, in_range);
      bit_mask_t old_mask = bitmask[mask_idx];

      if (lane_id == 0) {
        bit_mask_t new_mask = (old_mask & ~active_mask) |
                              ((value_is_valid ? 0xFFFFFFFF : 0) & active_mask);
        bitmask[mask_idx] = new_mask;
        // null_diff = (mask_size - __popc(new_mask)) - (mask_size - __popc(old_mask))
        atomicAdd(null_count, __popc(old_mask) - __popc(new_mask));
      }
    }

    mask_idx += masks_per_grid;
  }
}

struct fill_dispatch {
  template <typename T>
  void operator()(gdf_column *column, gdf_scalar const& value, 
                  gdf_index_type begin, gdf_index_type end,
                  cudaStream_t stream = 0)
  {
    static_assert(warp_size == cudf::util::size_in_bits<bit_mask_t>(), 
      "fill_kernel assumes bitmask element size in bits == warp size");

    auto fill = fill_kernel<T, false>;
    gdf_size_type *null_count = nullptr;

    if (column->valid != nullptr) {
      RMM_ALLOC(&null_count, sizeof(gdf_size_type), stream);
      CUDA_TRY(cudaMemsetAsync(null_count, column->null_count, 
                               sizeof(gdf_size_type), stream));
      fill = fill_kernel<T, true>;
    }

    // This one results in a compiler internal error! TODO: file bug
    //gdf_size_type num_items = cudf::util::round_up_safe(end - begin, warp_size);
    // number threads to cover range, rounded to nearest warp
    gdf_size_type num_items =
      warp_size * cudf::util::div_rounding_up_safe(end - begin, warp_size);

    constexpr int block_size = 256;

    cudf::util::cuda::grid_config_1d grid{num_items, block_size, 1};

    T * __restrict__ data = static_cast<T*>(column->data);
    bit_mask_t * __restrict__ bitmask =
      reinterpret_cast<bit_mask_t*>(column->valid);
    T const val = *reinterpret_cast<T const*>(&value.data);

    fill<<<grid.num_blocks, block_size, 0, stream>>>(data, bitmask, null_count,
                                                     begin, end,
                                                     val, value.is_valid);

    if (column->valid != nullptr) {
      CUDA_TRY(cudaMemcpyAsync(&column->null_count, null_count,
                               sizeof(gdf_size_type), cudaMemcpyDefault, stream));
      RMM_FREE(null_count, stream);
    }
    //thrust::fill(rmm::exec_policy(stream)->on(stream),
    //             data + begin, data + end, val);
    CHECK_STREAM(stream);
  }
};

}; // namespace anonymous

namespace cudf {

void fill(gdf_column *column, gdf_scalar const& value, 
          gdf_index_type begin, gdf_index_type end)
{
  CUDF_EXPECTS(column != nullptr, "Column is null");
  CUDF_EXPECTS(column->data != nullptr, "Data pointer is null");
  CUDF_EXPECTS(column->dtype == value.dtype, "Data type mismatch");

  cudf::type_dispatcher(column->dtype,
                        fill_dispatch{},
                        column, value, begin, end);
}

}; // namespace cudf