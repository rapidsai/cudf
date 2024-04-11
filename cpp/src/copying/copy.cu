/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>

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
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::column_view>)>
  auto operator()(T const& input, rmm::cuda_stream_view stream)
  {
    return cudf::column_device_view::create(input, stream);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::scalar>)>
  auto operator()(T const& input, rmm::cuda_stream_view)
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
    auto p_lhs      = get_iterable_device_view{}(lhs_h, stream);
    auto p_rhs      = get_iterable_device_view{}(rhs_h, stream);
    auto const& lhs = *p_lhs;
    auto const& rhs = *p_rhs;

    auto lhs_iter = cudf::detail::make_optional_iterator<T>(lhs, nullate::DYNAMIC{left_nullable});
    auto rhs_iter = cudf::detail::make_optional_iterator<T>(rhs, nullate::DYNAMIC{right_nullable});
    return detail::copy_if_else(left_nullable || right_nullable,
                                lhs_iter,
                                lhs_iter + size,
                                rhs_iter,
                                filter,
                                lhs.type(),
                                stream,
                                mr);
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

    auto p_lhs      = get_iterable_device_view{}(lhs_h, stream);
    auto p_rhs      = get_iterable_device_view{}(rhs_h, stream);
    auto const& lhs = *p_lhs;
    auto const& rhs = *p_rhs;

    auto lhs_iter = cudf::detail::make_optional_iterator<T>(lhs, nullate::DYNAMIC{left_nullable});
    auto rhs_iter = cudf::detail::make_optional_iterator<T>(rhs, nullate::DYNAMIC{right_nullable});
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
 * Handles nested-typed column views. Uses the iterator `is_left` to decide what row to pick for
 * the output column.
 *
 * Uses `rhs` as the destination for scatter. First gathers indices of rows to copy from lhs.
 *
 * @tparam Filter Bool iterator producing `true` for indices of output rows to copy from `lhs` and
 * `false` for indices of output rows to copy from `rhs`
 * @param lhs Left-hand side input column view
 * @param rhs Right-hand side input column view
 * @param size The size of the output column, inputs rows are iterated from 0 to `size - 1`
 * @param is_left Predicate for picking rows from `lhs` on `true` or `rhs` on `false`
 * @param stream The stream on which to perform the allocation
 * @param mr The resource used to allocate the device storage
 * @return Column with rows populated according to the `is_left` predicate
 */
template <typename Filter>
std::unique_ptr<column> scatter_gather_based_if_else(cudf::column_view const& lhs,
                                                     cudf::column_view const& rhs,
                                                     size_type size,
                                                     Filter is_left,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  auto gather_map = rmm::device_uvector<size_type>{static_cast<std::size_t>(size), stream};
  auto const gather_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                              thrust::make_counting_iterator(size_type{0}),
                                              thrust::make_counting_iterator(size_type{size}),
                                              gather_map.begin(),
                                              is_left);

  gather_map.resize(thrust::distance(gather_map.begin(), gather_map_end), stream);

  auto const scatter_src_lhs = cudf::detail::gather(table_view{std::vector<column_view>{lhs}},
                                                    gather_map,
                                                    out_of_bounds_policy::DONT_CHECK,
                                                    negative_index_policy::NOT_ALLOWED,
                                                    stream,
                                                    rmm::mr::get_current_device_resource());

  auto result = cudf::detail::scatter(
    table_view{std::vector<column_view>{scatter_src_lhs->get_column(0).view()}},
    gather_map,
    table_view{std::vector<column_view>{rhs}},
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

  auto const scatter_map_size  = std::distance(scatter_map.begin(), scatter_map_end);
  auto scatter_source          = std::vector<std::reference_wrapper<scalar const>>{std::ref(lhs)};
  auto scatter_map_column_view = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(scatter_map_size),
                                                   scatter_map.begin(),
                                                   nullptr,
                                                   0};

  auto result = cudf::detail::scatter(
    scatter_source, scatter_map_column_view, table_view{std::vector<column_view>{rhs}}, stream, mr);

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

template <>
struct copy_if_else_functor_impl<struct_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool,
                                     bool,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return scatter_gather_based_if_else(lhs, rhs, size, filter, stream, mr);
  }
};

template <>
struct copy_if_else_functor_impl<list_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool,
                                     bool,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return scatter_gather_based_if_else(lhs, rhs, size, filter, stream, mr);
  }
};

template <>
struct copy_if_else_functor_impl<dictionary32> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool,
                                     bool,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return scatter_gather_based_if_else(lhs, rhs, size, filter, stream, mr);
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
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
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
  CUDF_EXPECTS(boolean_mask.type() == data_type(type_id::BOOL8),
               "Boolean mask column must be of type type_id::BOOL8",
               cudf::data_type_error);

  if (boolean_mask.is_empty()) { return cudf::empty_like(lhs); }

  auto bool_mask_device_p             = column_device_view::create(boolean_mask, stream);
  column_device_view bool_mask_device = *bool_mask_device_p;

  auto const has_nulls = boolean_mask.has_nulls();
  auto filter          = [bool_mask_device, has_nulls] __device__(cudf::size_type i) {
    return (!has_nulls || bool_mask_device.is_valid_nocheck(i)) and
           bool_mask_device.element<bool>(i);
  };

  // always dispatch on dictionary-type if either input is a dictionary
  auto dispatch_type = cudf::is_dictionary(rhs.type()) ? rhs.type() : lhs.type();

  return cudf::type_dispatcher<dispatch_storage_type>(dispatch_type,
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

};  // namespace

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs and rhs columns",
               std::invalid_argument);
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size", std::invalid_argument);
  CUDF_EXPECTS(
    lhs.type() == rhs.type(), "Both inputs must be of the same type", cudf::data_type_error);

  return copy_if_else(lhs, rhs, lhs.has_nulls(), rhs.has_nulls(), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == rhs.size(),
               "Boolean mask column must be the same size as rhs column",
               std::invalid_argument);

  auto rhs_type =
    cudf::is_dictionary(rhs.type()) ? cudf::dictionary_column_view(rhs).keys_type() : rhs.type();
  CUDF_EXPECTS(
    lhs.type() == rhs_type, "Both inputs must be of the same type", cudf::data_type_error);

  return copy_if_else(lhs, rhs, !lhs.is_valid(stream), rhs.has_nulls(), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs column",
               std::invalid_argument);

  auto lhs_type =
    cudf::is_dictionary(lhs.type()) ? cudf::dictionary_column_view(lhs).keys_type() : lhs.type();
  CUDF_EXPECTS(
    lhs_type == rhs.type(), "Both inputs must be of the same type", cudf::data_type_error);

  return copy_if_else(lhs, rhs, lhs.has_nulls(), !rhs.is_valid(stream), boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    lhs.type() == rhs.type(), "Both inputs must be of the same type", cudf::data_type_error);
  return copy_if_else(
    lhs, rhs, !lhs.is_valid(stream), !rhs.is_valid(stream), boolean_mask, stream, mr);
}

};  // namespace detail

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, stream, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, stream, mr);
}

}  // namespace cudf
