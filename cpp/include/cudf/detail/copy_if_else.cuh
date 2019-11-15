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

#pragma once

#include <cudf/cudf.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <cub/cub.cuh>


namespace cudf {
namespace experimental {
namespace detail {

namespace {    // anonymous

template <size_type block_size, typename T, typename Filter, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_if_else_kernel(  column_device_view const lhs,
                           column_device_view const rhs,
                           Filter filter,
                           mutable_column_device_view out,
                           size_type * __restrict__ const valid_count)
{   
   const size_type tid = threadIdx.x + blockIdx.x * blockDim.x;
   const int w_id = tid / warp_size;
   const size_type warps_per_grid = gridDim.x * blockDim.x / warp_size;
   // begin/end indices for the column data
   size_type begin = 0;
   size_type end = lhs.size();   
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via 
   // __ballot_sync()
   size_type w_begin = cudf::util::detail::bit_container_index<bit_mask::bit_mask_t>(begin);
   size_type w_end = cudf::util::detail::bit_container_index<bit_mask::bit_mask_t>(end);

   // lane id within the current warp
   constexpr size_type w_leader_lane{0};
   const int w_lane_id = threadIdx.x % warp_size;

   size_type w_valid_count{0};
  
   // current warp.
   size_type w_cur = w_begin + w_id;
   size_type index = tid;
   while(w_cur <= w_end){
      bool in_range = (index >= begin && index < end);

      bool valid = true;
      if(has_validity){
         valid = in_range && filter(index) ? lhs.is_valid(index) : rhs.is_valid(index);
      }

      // do the copy if-else
      if(in_range){ 
         out.element<T>(index) = filter(index) ? lhs.element<T>(index) : rhs.element<T>(index);
      }

      // update validity
      if(has_validity){
         // the final validity mask for this warp
         int w_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
         // only one guy in the warp needs to update the mask and count
         if(w_lane_id == 0){
            out.set_mask_word(w_cur, w_mask);
            w_valid_count += __popc(w_mask);
         }
      }

      // next grid
      w_cur += warps_per_grid;
      index += blockDim.x * gridDim.x;      
   }

   if(has_validity){
      // sum all null counts across all warps
      size_type b_valid_count = single_lane_block_sum_reduce<block_size, w_leader_lane>(w_valid_count);
      // b_null_count will only be valid on thread 0
      if(threadIdx.x == 0){
         // using an atomic here because there are multiple blocks doing this work
         atomicAdd(valid_count, b_valid_count);
      }      
   }
}



/* --------------------------------------------------------------------------*/
/**
* @brief Functor called by the `type_dispatcher` in order to perform a copy if/else
*        using a filter function to select from lhs/rhs columns.
*/
/* ----------------------------------------------------------------------------*/
struct copy_if_else_functor {
   template <typename T, typename Filter>
   std::unique_ptr<column> operator()(column_view const& lhs,
                                    column_view const& rhs,
                                    Filter filter,
                                    rmm::mr::device_memory_resource *mr,
                                    cudaStream_t stream)
   {
      // output
      std::unique_ptr<column> out = experimental::allocate_like(lhs, lhs.size(), experimental::mask_allocation_policy::RETAIN, mr);

      cudf::size_type num_els = cudf::util::round_up_safe(lhs.size(), warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};

      // device views
      auto lhs_dv = column_device_view::create(lhs);
      auto rhs_dv = column_device_view::create(rhs);
      auto out_dv = mutable_column_device_view::create(*out);
      
      if(lhs.nullable()){                           
         rmm::device_scalar<cudf::size_type> valid_count{0, stream, mr};
         
         // call the kernel
         copy_if_else_kernel<block_size, T, Filter, true><<<grid.num_blocks, block_size, 0, stream>>>(
            *lhs_dv, *rhs_dv, filter, *out_dv, valid_count.data());

         out->set_null_count(lhs.size() - valid_count.value());
      } else {
         // call the kernel
         copy_if_else_kernel<block_size, T, Filter, false><<<grid.num_blocks, block_size, 0, stream>>>(
            *lhs_dv, *rhs_dv, filter, *out_dv, nullptr);
      }

      return out;
   }
};

}  // anonymous namespace

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or 
 *          @p rhs based on the filter lambda. 
 * 
 * @p filter must be a functor or lambda with the following signature:
 * bool __device__ operator()(cudf::size_type i);
 * It should return true if element i of @p lhs should be selected, or false if element i of @p rhs should be selected. 
 *         
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length  
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] filter lambda. 
 * @param[in] mr resource for allocating device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns new column with the selected elements
 */
template<typename Filter>
std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, Filter filter,
                                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{         
   return cudf::experimental::type_dispatcher(lhs.type(),
                                             copy_if_else_functor{},
                                             lhs,
                                             rhs,
                                             filter,
                                             mr,
                                             stream);
}

}  // namespace detail

}  // namespace experimental

}  // namespace cudf
