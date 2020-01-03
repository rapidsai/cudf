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

#include <cudf/wrappers/bool.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace experimental {

namespace {

template <typename Left, typename Right, typename Filter>
struct copy_if_else_functor {     
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()(Left const& lhs, Right const& rhs,                                      
                                      bool left_nullable, bool right_nullable,
                                      Filter filter,
                                      rmm::mr::device_memory_resource* mr)
   { 
      return cudf::make_empty_column(lhs.type());
   }

   template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
   std::unique_ptr<column> operator()(Left const& lhs, Right const& rhs,                                      
                                      bool left_nullable, bool right_nullable,
                                      Filter filter,
                                      rmm::mr::device_memory_resource* mr)
   {                            
      if(left_nullable){
         if(right_nullable){
            auto lhs_iter = make_pair_iterator<true, T>(lhs.template data<T>(), lhs.null_mask());
            auto rhs_iter = make_pair_iterator<true, T>(rhs.template data<T>(), rhs.null_mask());
            return detail::copy_if_else<T>(lhs.type(), true, lhs_iter, lhs_iter + lhs.size(), rhs_iter,
                                          filter, mr, (cudaStream_t)0);
         }
         auto lhs_iter = make_pair_iterator<true, T>(lhs.template data<T>(), lhs.null_mask());
         auto rhs_iter = make_pair_iterator<false, T>(rhs.template data<T>(), rhs.null_mask());
         return detail::copy_if_else<T>(lhs.type(), true, lhs_iter, lhs_iter + lhs.size(), rhs_iter,                                      
                                    filter, mr, (cudaStream_t)0);
      }
      if(right_nullable){
         auto lhs_iter = make_pair_iterator<false, T>(lhs.template data<T>(), lhs.null_mask());
         auto rhs_iter = make_pair_iterator<true, T>(rhs.template data<T>(), rhs.null_mask());
         return detail::copy_if_else<T>(lhs.type(), true, lhs_iter, lhs_iter + lhs.size(), rhs_iter,                                      
                                    filter, mr, (cudaStream_t)0);
      }     
      auto lhs_iter = make_pair_iterator<false, T>(lhs.template data<T>(), lhs.null_mask());
      auto rhs_iter = make_pair_iterator<false, T>(rhs.template data<T>(), rhs.null_mask());
      return detail::copy_if_else<T>(lhs.type(), false, lhs_iter, lhs_iter + lhs.size(), rhs_iter,
                                 filter, mr, (cudaStream_t)0);
   }
};

// wrap up boolean_mask into a filter lambda
template <typename Left, typename Right>
std::unique_ptr<column> copy_if_else( Left const& lhs,
                                      Right const& rhs,
                                      bool left_nullable, bool right_nullable,
                                      column_view const& boolean_mask, bool invert_mask,                                      
                                      rmm::mr::device_memory_resource* mr)
{   
   CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
   CUDF_EXPECTS(not boolean_mask.has_nulls(), "Boolean mask must not contain null values.");
   CUDF_EXPECTS(boolean_mask.type() == data_type(BOOL8), "Boolean mask column must be of type BOOL8");

   if (boolean_mask.size() == 0) {
      return cudf::make_empty_column(lhs.type());
   }

   auto bool_mask_device_p = column_device_view::create(boolean_mask);
   column_device_view bool_mask_device = *bool_mask_device_p;                                    

   if(invert_mask){
      auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return !bool_mask_device.element<cudf::experimental::bool8>(i); };
      return cudf::experimental::type_dispatcher(lhs.type(),
                                                copy_if_else_functor<Left, Right, decltype(filter)>{},
                                                lhs, rhs,
                                                left_nullable, right_nullable,
                                                filter,
                                                mr);
   }
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); };
   return cudf::experimental::type_dispatcher(lhs.type(),
                                             copy_if_else_functor<Left, Right, decltype(filter)>{},
                                             lhs, rhs,
                                             left_nullable, right_nullable,
                                             filter,
                                             mr);
}

}; // namespace anonymous


std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource* mr)
{      
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs and rhs columns");      
   CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size"); 
   return copy_if_else(lhs, rhs, lhs.nullable(), rhs.nullable(), boolean_mask, false, mr);
}

std::unique_ptr<column> copy_if_else( scalar const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource* mr)
{   
   CUDF_EXPECTS(boolean_mask.size() == rhs.size(), "Boolean mask column must be the same size as rhs column");
   // return detail::copy_if_else(lhs, rhs, boolean_mask, false, mr, (cudaStream_t)0);
    return cudf::make_empty_column(lhs.type());
}

std::unique_ptr<column> copy_if_else( column_view const& lhs, scalar const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource* mr)
{
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs column");
   // return detail::copy_if_else(rhs, lhs, boolean_mask, true, mr, (cudaStream_t)0); 
   return cudf::make_empty_column(lhs.type());
}

std::unique_ptr<column> copy_if_else( scalar const& lhs, scalar const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource* mr)
{
    // return detail::copy_if_else(lhs, rhs, boolean_mask, false, mr, (cudaStream_t)0); 
    return cudf::make_empty_column(lhs.type());
}

} // namespace experimental

} // namespace cudf
