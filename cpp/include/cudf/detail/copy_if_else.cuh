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

#include <thrust/transform.h>
#include <rmm/thrust_rmm_allocator.h>

#include <cudf/cudf.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>


namespace cudf {
namespace experimental {
namespace detail {

namespace {    // anonymous

/* --------------------------------------------------------------------------*/
/**
* @brief Functor called by the `type_dispatcher` in order to perform a copy if/else
*        using a filter function to select from lhs/rhs columns.
*/
/* ----------------------------------------------------------------------------*/
struct copy_if_else_functor {
   template <typename T, typename Filter>
   void operator()(  Filter filter,
                     column_view const& lhs,
                     column_view const& rhs,
                     mutable_column_view& out,
                     cudaStream_t stream)
   {
      auto begin  = thrust::make_zip_iterator(thrust::make_tuple( thrust::make_counting_iterator(0),
                                                                  lhs.begin<T>(),
                                                                  rhs.begin<T>()));

      auto end  = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(lhs.size()),
                                                               lhs.end<T>(),
                                                               rhs.end<T>()));
      
      thrust::transform(rmm::exec_policy(stream)->on(stream), begin, end, out.begin<T>(),
                        [filter] __device__ (thrust::tuple<size_type, T, T> i)
                        {
                           return filter(thrust::get<0>(i)) ? thrust::get<1>(i) : thrust::get<2>(i);
                        });
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
 * @param[in] filter lambda. 
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] mr resource for allocating device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns new column with the selected elements
 */
template<typename Filter>
std::unique_ptr<column> copy_if_else( Filter filter, column_view const& lhs, column_view const& rhs,
                                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
   // output
   std::unique_ptr<column> out = experimental::allocate_like(lhs, lhs.size(), experimental::mask_allocation_policy::RETAIN, mr);
   auto mutable_view = out->mutable_view();
   
   cudf::experimental::type_dispatcher(lhs.type(), 
                                       copy_if_else_functor{},
                                       filter,
                                       lhs,
                                       rhs,
                                       mutable_view,
                                       stream);

   return out;
}

}  // namespace detail

}  // namespace experimental

}  // namespace cudf
