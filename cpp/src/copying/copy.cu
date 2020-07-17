/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Specialization of copy_if_else_functor for string_views.
 */
template <typename T, typename Left, typename Right, typename Filter>
struct copy_if_else_functor_impl {
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
        return detail::copy_if_else(true, lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
      return detail::copy_if_else(true, lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
      return detail::copy_if_else(true, lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
    return detail::copy_if_else(false, lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
  }
};

/**
 * @brief Specialization of copy_if_else_functor for string_views.
 */
template <typename Left, typename Right, typename Filter>
struct copy_if_else_functor_impl<string_view, Left, Right, Filter> {
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    using T = string_view;

    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
        return strings::detail::copy_if_else(
          lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
    return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, mr, stream);
  }
};

/**
 * @brief Specialization of copy_if_else_functor for list_views.
 */
template <typename Left, typename Right, typename Filter>
struct copy_if_else_functor_impl<list_view, Left, Right, Filter> {
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("copy_if_else not supported for list_view yet");
  }
};

/**
 * @brief Functor called by the `type_dispatcher` to invoke copy_if_else on combinations
 *        of column_view and scalar
 */
struct copy_if_else_functor {
  template <typename T, typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    copy_if_else_functor_impl<T, Left, Right, Filter> copier{};
    return copier(lhs, rhs, size, left_nullable, right_nullable, filter, mr, stream);
  }
};

// wrap up boolean_mask into a filter lambda
template <typename Left, typename Right>
std::unique_ptr<column> copy_if_else(Left const& lhs,
                                     Right const& rhs,
                                     bool left_nullable,
                                     bool right_nullable,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
  CUDF_EXPECTS(boolean_mask.type() == data_type(type_id::BOOL8),
               "Boolean mask column must be of type type_id::BOOL8");

  if (boolean_mask.size() == 0) { return cudf::make_empty_column(lhs.type()); }

  auto bool_mask_device_p             = column_device_view::create(boolean_mask);
  column_device_view bool_mask_device = *bool_mask_device_p;

  if (boolean_mask.has_nulls()) {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.is_valid_nocheck(i) and bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher(lhs.type(),
                                 copy_if_else_functor{},
                                 lhs,
                                 rhs,
                                 boolean_mask.size(),
                                 left_nullable,
                                 right_nullable,
                                 filter,
                                 mr,
                                 stream);
  } else {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher(lhs.type(),
                                 copy_if_else_functor{},
                                 lhs,
                                 rhs,
                                 boolean_mask.size(),
                                 left_nullable,
                                 right_nullable,
                                 filter,
                                 mr,
                                 stream);
  }
}

};  // namespace

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs and rhs columns");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size");
  return copy_if_else(*column_device_view::create(lhs),
                      *column_device_view::create(rhs),
                      lhs.has_nulls(),
                      rhs.has_nulls(),
                      boolean_mask,
                      mr,
                      stream);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  CUDF_EXPECTS(boolean_mask.size() == rhs.size(),
               "Boolean mask column must be the same size as rhs column");
  return copy_if_else(lhs,
                      *column_device_view::create(rhs),
                      !lhs.is_valid(),
                      rhs.has_nulls(),
                      boolean_mask,
                      mr,
                      stream);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs column");
  return copy_if_else(*column_device_view::create(lhs),
                      rhs,
                      lhs.has_nulls(),
                      !rhs.is_valid(),
                      boolean_mask,
                      mr,
                      stream);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
{
  return copy_if_else(lhs, rhs, !lhs.is_valid(), !rhs.is_valid(), boolean_mask, mr, stream);
}

};  // namespace detail

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, mr);
}

}  // namespace cudf
