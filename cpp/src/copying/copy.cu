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

namespace cudf {
namespace experimental {
namespace detail {

// column/column, scalar/column
template <typename Left, typename Right>
std::unique_ptr<column> copy_if_else( Left const& lhs, Right const& rhs, 
                                      column_view const& boolean_mask, bool invert_mask,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream)
{
   CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
   CUDF_EXPECTS(not boolean_mask.has_nulls(), "Boolean mask must not contain null values.");
   CUDF_EXPECTS(boolean_mask.type() == data_type(BOOL8), "Boolean mask column must be of type BOOL8");   
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs and rhs columns");
   if (lhs.size() == 0) {
       return cudf::experimental::empty_like(lhs);
   }

   auto bool_mask_device_p = column_device_view::create(boolean_mask);
   column_device_view bool_mask_device = *bool_mask_device_p;

   if(invert_mask){
      auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return !bool_mask_device.element<cudf::experimental::bool8>(i); };
      return copy_if_else(lhs, rhs, filter, mr, stream);
   }
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); };
   return copy_if_else(lhs, rhs, filter, mr, stream);
}

// scalar/scalar.
std::unique_ptr<column> copy_if_else( scalar const& lhs, scalar const& rhs,
                                      column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream)
{
   CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
   CUDF_EXPECTS(not boolean_mask.has_nulls(), "Boolean mask must not contain null values.");
   CUDF_EXPECTS(boolean_mask.type() == data_type(BOOL8), "Boolean mask column must be of type BOOL8");

   auto bool_mask_device_p = column_device_view::create(boolean_mask);
   column_device_view bool_mask_device = *bool_mask_device_p;
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); };

   return copy_if_else(lhs, rhs, boolean_mask.size(), filter, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr)
{
   CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size");
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs and rhs columns");
   return detail::copy_if_else(lhs, rhs, boolean_mask, false, mr, 0);
}

std::unique_ptr<column> copy_if_else( scalar const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr)
{
   CUDF_EXPECTS(boolean_mask.size() == rhs.size(), "Boolean mask column must be the same size as rhs column"); 
   return detail::copy_if_else(lhs, rhs, boolean_mask, false, mr, 0);
}

std::unique_ptr<column> copy_if_else( column_view const& lhs, scalar const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr)
{
   CUDF_EXPECTS(boolean_mask.size() == lhs.size(), "Boolean mask column must be the same size as lhs column");
   return detail::copy_if_else(rhs, lhs, boolean_mask, true, mr, 0);
}

std::unique_ptr<column> copy_if_else( scalar const& lhs, scalar const& rhs, column_view const& boolean_mask,
                                      rmm::mr::device_memory_resource *mr)
{
   return detail::copy_if_else(lhs, rhs, boolean_mask, mr, 0);
}

} // namespace experimental
} // namespace cudf
