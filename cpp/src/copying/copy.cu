/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
namespace {

template <typename T, typename Enable = void>
struct copy_if_else_functor_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for copy_if_else.");
  }
};

/**
 * @brief Functor to fetch a device-view for the specified scalar/column_view.
 */
struct get_iterable_device_view {
  template <typename T, CUDF_ENABLE_IF(std::is_same<T, cudf::column_view>::value)>
  auto operator()(T const& input)
  {
    return cudf::column_device_view::create(input);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same<T, cudf::scalar>::value)>
  auto operator()(T const& input)
  {
    return &input;
  }
};

template <typename T>
struct copy_if_else_functor_impl<T, std::enable_if_t<is_rep_layout_compatible<T>()>> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs_h,
                                     Right const& rhs_h,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto p_lhs      = get_iterable_device_view{}(lhs_h);
    auto p_rhs      = get_iterable_device_view{}(rhs_h);
    auto const& lhs = *p_lhs;
    auto const& rhs = *p_rhs;

    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
        return detail::copy_if_else(
          true, lhs_iter, lhs_iter + size, rhs_iter, filter, lhs.type(), stream, mr);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
      return detail::copy_if_else(
        true, lhs_iter, lhs_iter + size, rhs_iter, filter, lhs.type(), stream, mr);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
      return detail::copy_if_else(
        true, lhs_iter, lhs_iter + size, rhs_iter, filter, lhs.type(), stream, mr);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
    return detail::copy_if_else(
      false, lhs_iter, lhs_iter + size, rhs_iter, filter, lhs.type(), stream, mr);
  }
};

/**
 * @brief Specialization of copy_if_else_functor for string_views.
 */
template <>
struct copy_if_else_functor_impl<string_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs_h,
                                     Right const& rhs_h,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    using T = string_view;

    auto p_lhs      = get_iterable_device_view{}(lhs_h);
    auto p_rhs      = get_iterable_device_view{}(rhs_h);
    auto const& lhs = *p_lhs;
    auto const& rhs = *p_rhs;

    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
        return strings::detail::copy_if_else(
          lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(rhs);
    return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
  }
};

/**
 * @brief Adapter to negate predicates.
 */
template <typename Predicate>
class logical_not {
 public:
  explicit logical_not(Predicate predicate) : _pred{predicate} {}
  bool __device__ operator()(size_type i) const { return not _pred(i); }

 private:
  Predicate _pred;
};

/**
 * @brief Implementation of copy_if_else() with gather()/scatter().
 *
 * Handles nested-typed columns and scalars
 */
template <typename Filter>
std::unique_ptr<column> scatter_gather_based_if_else(cudf::column_view const& lhs,
                                                     cudf::column_view const& rhs,
                                                     size_type size,
                                                     Filter is_left,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  auto scatter_map = rmm::device_uvector<size_type>{static_cast<std::size_t>(size), stream};
  auto const scatter_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                               thrust::make_counting_iterator(size_type{0}),
                                               thrust::make_counting_iterator(size_type{size}),
                                               scatter_map.begin(),
                                               is_left);

  auto const scatter_src_lhs = cudf::detail::gather(table_view{std::vector<column_view>{lhs}},
                                                    scatter_map.begin(),
                                                    scatter_map_end,
                                                    out_of_bounds_policy::DONT_CHECK,
                                                    stream);

  auto result = cudf::detail::scatter(
    table_view{std::vector<column_view>{scatter_src_lhs->get_column(0).view()}},
    scatter_map.begin(),
    scatter_map_end,
    table_view{std::vector<column_view>{rhs}},
    false,
    stream,
    mr);

  return std::move(result->release()[0]);
}

template <typename Filter>
std::unique_ptr<column> scatter_gather_based_if_else(cudf::scalar const& lhs,
                                                     cudf::column_view const& rhs,
                                                     size_type size,
                                                     Filter is_left,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  auto scatter_map = rmm::device_uvector<size_type>{static_cast<std::size_t>(size), stream};
  auto const scatter_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                               thrust::make_counting_iterator(size_type{0}),
                                               thrust::make_counting_iterator(size_type{size}),
                                               scatter_map.begin(),
                                               is_left);

  auto const scatter_map_size  = scatter_map_end - scatter_map.begin();
  auto scatter_source          = std::vector<std::reference_wrapper<const scalar>>{std::ref(lhs)};
  auto scatter_map_column_view = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(scatter_map_size),
                                                   scatter_map.begin()};

  auto result = cudf::scatter(
    scatter_source, scatter_map_column_view, table_view{std::vector<column_view>{rhs}}, false, mr);

  return std::move(result->release()[0]);
}

template <typename Filter>
std::unique_ptr<column> scatter_gather_based_if_else(cudf::column_view const& lhs,
                                                     cudf::scalar const& rhs,
                                                     size_type size,
                                                     Filter is_left,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  return scatter_gather_based_if_else(rhs, lhs, size, logical_not{is_left}, stream, mr);
}

template <typename Filter>
std::unique_ptr<column> scatter_gather_based_if_else(cudf::scalar const& lhs,
                                                     cudf::scalar const& rhs,
                                                     size_type size,
                                                     Filter is_left,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  auto rhs_col = cudf::make_column_from_scalar(rhs, size, stream, mr);
  return scatter_gather_based_if_else(lhs, rhs_col->view(), size, is_left, stream, mr);
}

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
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if constexpr (std::is_same_v<T, cudf::list_view> or std::is_same_v<T, cudf::struct_view>) {
      (void)left_nullable;
      (void)right_nullable;
      return scatter_gather_based_if_else(lhs, rhs, size, filter, stream, mr);
    }

    copy_if_else_functor_impl<T> copier{};
    return copier(lhs, rhs, size, left_nullable, right_nullable, filter, stream, mr);
  }
};

// wrap up boolean_mask into a filter lambda
template <typename Left, typename Right>
std::unique_ptr<column> copy_if_else(Left const& lhs,
                                     Right const& rhs,
                                     bool left_nullable,
                                     bool right_nullable,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
  CUDF_EXPECTS(boolean_mask.type() == data_type(type_id::BOOL8),
               "Boolean mask column must be of type type_id::BOOL8");

  if (boolean_mask.is_empty()) { return cudf::empty_like(lhs); }

  auto bool_mask_device_p             = column_device_view::create(boolean_mask);
  column_device_view bool_mask_device = *bool_mask_device_p;

  if (boolean_mask.has_nulls()) {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.is_valid_nocheck(i) and bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher<dispatch_storage_type>(lhs.type(),
                                                        copy_if_else_functor{},
                                                        lhs,
                                                        rhs,
                                                        boolean_mask.size(),
                                                        left_nullable,
                                                        right_nullable,
                                                        filter,
                                                        stream,
                                                        mr);
  } else {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher<dispatch_storage_type>(lhs.type(),
                                                        copy_if_else_functor{},
                                                        lhs,
                                                        rhs,
                                                        boolean_mask.size(),
                                                        left_nullable,
                                                        right_nullable,
                                                        filter,
                                                        stream,
                                                        mr);
  }
}

};  // namespace

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs and rhs columns");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size");
  return copy_if_else(lhs, rhs, lhs.has_nulls(), rhs.has_nulls(), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == rhs.size(),
               "Boolean mask column must be the same size as rhs column");
  return copy_if_else(lhs, rhs, !lhs.is_valid(), rhs.has_nulls(), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs column");
  return copy_if_else(lhs, rhs, lhs.has_nulls(), !rhs.is_valid(), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return copy_if_else(lhs, rhs, !lhs.is_valid(), !rhs.is_valid(), boolean_mask, stream, mr);
}

};  // namespace detail

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
