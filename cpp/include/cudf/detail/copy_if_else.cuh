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
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>
#include <iterator/legacy/iterator.cuh>

#include <rmm/device_scalar.hpp>
#include <cub/cub.cuh>


namespace cudf {
namespace experimental {
namespace detail {

namespace {    // anonymous

template <size_type block_size, typename T, typename LeftIter, typename RightIter, typename Filter, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_if_else_kernel(  LeftIter lhs,
                           RightIter rhs,                           
                           Filter filter,
                           mutable_column_device_view out,
                           size_type * __restrict__ const valid_count)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / warp_size;

   // begin/end indices for the column data
   size_type begin = 0;
   size_type end = out.size();   
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via
   // __ballot_sync()
   size_type warp_begin = cudf::word_index(begin);
   size_type warp_end = cudf::word_index(end-1);

   // lane id within the current warp
   constexpr size_type leader_lane{0};
   const int lane_id = threadIdx.x % warp_size;

   size_type warp_valid_count{0};

   // current warp.
   size_type warp_cur = warp_begin + warp_id;
   size_type index = tid;
   while(warp_cur <= warp_end){
      bool in_range = (index >= begin && index < end);

      bool valid = true;
      if(has_validity){
         valid = in_range && (filter(index) ? thrust::get<1>(lhs[index]) : thrust::get<1>(rhs[index]));
      }      

      // do the copy if-else
      if(in_range){          
         out.element<T>(index) = filter(index) ?   static_cast<T>(thrust::get<0>(lhs[index])) :
                                                   static_cast<T>(thrust::get<0>(rhs[index]));
      }

      // update validity
      if(has_validity){
         // the final validity mask for this warp
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
         // only one guy in the warp needs to update the mask and count
         if(lane_id == 0){
            out.set_mask_word(warp_cur, warp_mask);
            warp_valid_count += __popc(warp_mask);
         }
      }

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }

   if(has_validity){
      // sum all null counts across all warps
      size_type block_valid_count = single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
      // block_valid_count will only be valid on thread 0
      if(threadIdx.x == 0){
         // using an atomic here because there are multiple blocks doing this work
         atomicAdd(valid_count, block_valid_count);
      }
   }
}

/**
* @brief Functor called by the `type_dispatcher` in order to perform a copy if/else
*        using a filter function to select from lhs/rhs columns.
*/
/* ----------------------------------------------------------------------------*/
struct copy_if_else_functor_impl {   
   template <typename Element, typename LeftIter, typename RightIter, typename FilterFn>
   void launch(LeftIter lhs,
               RightIter rhs, 
               FilterFn filter, 
               mutable_column_device_view & out,
               cudf::size_type & out_null_count,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
   {
      cudf::size_type num_els = cudf::util::round_up_safe(out.size(), warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};

      // if we have validity in the output
      if(out.nullable()){
         rmm::device_scalar<cudf::size_type> valid_count{0, stream, mr};

         // call the kernel
         copy_if_else_kernel<block_size, Element, LeftIter, RightIter, FilterFn, true><<<grid.num_blocks, block_size, 0, stream>>>(
            lhs, rhs, filter, out, valid_count.data());

         out_null_count = out.size() - valid_count.value();
      } else {                  
         // call the kernel
         copy_if_else_kernel<block_size, Element, LeftIter, RightIter, FilterFn, false><<<grid.num_blocks, block_size, 0, stream>>>(
            lhs, rhs, filter, out, nullptr);

         out_null_count = 0;
      }
   }
   
   // column/column
   template <typename T, typename Filter, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( column_view const &lhs, 
                                       column_view const &rhs,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {              
      // output
      auto validity_policy = lhs.has_nulls() or rhs.has_nulls() ? experimental::mask_allocation_policy::ALWAYS : experimental::mask_allocation_policy::NEVER;
      std::unique_ptr<column> out = experimental::allocate_like(lhs, lhs.size(), validity_policy, mr);
      auto out_dv = mutable_column_device_view::create(*out);      
      cudf::size_type out_null_count = 0;
      
      if(lhs.has_nulls()){
         if(rhs.has_nulls()){
            launch<T>(  make_pair_iterator<true, T>((T*)lhs.begin<T>(), lhs.null_mask()),
                        make_pair_iterator<true, T>((T*)rhs.begin<T>(), rhs.null_mask()),
                        filter, *out_dv, out_null_count, mr, stream);
         } else {
            launch<T>(  make_pair_iterator<true, T>((T*)lhs.begin<T>(), lhs.null_mask()),
                        make_pair_iterator<false, T>((T*)rhs.begin<T>(), static_cast<bitmask_type*>(nullptr)),
                        filter, *out_dv, out_null_count, mr, stream);
         }
      } else {            
         if(rhs.has_nulls()){
            launch<T>(  make_pair_iterator<false, T>((T*)lhs.begin<T>(), static_cast<bitmask_type*>(nullptr)),
                        make_pair_iterator<true, T>((T*)rhs.begin<T>(), rhs.null_mask()),
                        filter, *out_dv, out_null_count, mr, stream);
         } else {
            launch<T>(  make_pair_iterator<false, T>((T*)lhs.begin<T>(), static_cast<bitmask_type*>(nullptr)),
                        make_pair_iterator<false, T>((T*)rhs.begin<T>(), static_cast<bitmask_type*>(nullptr)),
                        filter, *out_dv, out_null_count, mr, stream);
         }
      }      
      
      out->set_null_count(out_null_count);

      return out;
   }       

   // scalar/column
   template <typename T, typename Filter, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( cudf::scalar const &lhs, 
                                       column_view const &rhs,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {                    
      // output
      auto validity_policy = rhs.has_nulls() ? experimental::mask_allocation_policy::ALWAYS : experimental::mask_allocation_policy::NEVER;
      std::unique_ptr<column> out = experimental::allocate_like(rhs, rhs.size(), validity_policy, mr);
      auto out_dv = mutable_column_device_view::create(*out);
      cudf::size_type out_null_count;            

      // data and validity iterators for scalar
      auto lhs_iter = thrust::make_constant_iterator(static_cast<cudf::experimental::scalar_type_t<T>const*>(&lhs)->value());
      auto lhs_valids = thrust::make_constant_iterator(1);      
      
      if(validity_policy == experimental::mask_allocation_policy::ALWAYS){
         launch<T>(  thrust::make_zip_iterator(thrust::make_tuple(lhs_iter, lhs_valids)),
                     make_pair_iterator<true, T>((T*)rhs.begin<T>(), rhs.null_mask()),
                     filter, *out_dv, out_null_count, mr, stream);
      } else {
         launch<T>(  thrust::make_zip_iterator(thrust::make_tuple(lhs_iter, lhs_valids)),
                     make_pair_iterator<false, T>((T*)rhs.begin<T>(), static_cast<bitmask_type*>(nullptr)),
                     filter, *out_dv, out_null_count, mr, stream);
      }

      out->set_null_count(out_null_count);                   

      return out;
   }
   
   /*
   // scalar/scalar   
   template <typename T, typename Filter, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( cudf::scalar const &lhs, 
                                       cudf::scalar const &rhs,
                                       size_type out_size,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {                            
      std::unique_ptr<column> out = experimental::allocate_like(lhs.type(), out_size, experimental::mask_allocation_policy::NEVER, mr);
      auto out_dv = mutable_column_device_view::create(*out);
      cudf::size_type out_null_count_unused;

      auto lhs_iter = thrust::make_constant_iterator(static_cast<cudf::experimental::scalar_type_t<T>const*>(&lhs)->value());
      auto lhs_valids = thrust::make_constant_iterator(1);      

      auto rhs_iter = thrust::make_constant_iterator(static_cast<cudf::experimental::scalar_type_t<T>const*>(&rhs)->value());
      auto rhs_valids = thrust::make_constant_iterator(1);      

      launch<T>(  thrust::make_zip_iterator(thrust::make_tuple(lhs_iter, lhs_valids)),
                  thrust::make_zip_iterator(thrust::make_tuple(rhs_iter, rhs_valids)),
                  filter, *out_dv, out_null_count_unused, mr, stream);      
   }      
   */

   // string_view overrides
   template <typename T, typename Filter, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( column_view const &lhs, 
                                       column_view const &rhs,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {
      CUDF_FAIL("Unxpected non fixed-width type");
   } 
   template <typename T, typename Filter, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( cudf::scalar const &lhs, 
                                       column_view const &rhs,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {
      CUDF_FAIL("Unxpected non fixed-width type");
   }
   template <typename T, typename Filter, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()( cudf::scalar const &lhs, 
                                       cudf::scalar const &rhs,
                                       Filter filter,                                    
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream)
   {
      CUDF_FAIL("Unxpected non fixed-width type");
   }  
};

/**
 * @brief Specialization functor for strings column to perform a copy if/else
 *        using a filter function to select from lhs/rhs columns.
 */
template<typename FilterFn>
struct copy_if_else_functor_impl<string_view, FilterFn>
{
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     column_view const& rhs,
                                     FilterFn filter,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream)
   {
      return strings::detail::copy_if_else( strings_column_view(lhs),
                                            strings_column_view(rhs),
                                            filter, mr, stream);
   }
};

/**
 * @brief Functor called by the `type_dispatcher` in order to perform a copy if/else
 *        using a filter function to select from lhs/rhs columns.
 *
 * This is required to split the specialization of the type Element from the Filter function.
 */
struct copy_if_else_functor
{
   template <typename Element, typename FilterFn>
   std::unique_ptr<column> operator()(column_view const& lhs,
                                      column_view const& rhs,
                                      FilterFn filter,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream)
   {
      copy_if_else_functor_impl<Element, FilterFn> copier{};
      return copier(lhs,rhs,filter,mr,stream);
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
template<typename FilterFn>
std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, FilterFn filter,
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

template<typename FilterFn>
std::unique_ptr<column> copy_if_else( scalar const& lhs, column_view const& rhs, FilterFn filter,
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
