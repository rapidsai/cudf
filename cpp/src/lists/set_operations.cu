/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/lists/set_operations.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief
 */
enum class operation_type { INTERSECTION, UNION, DIFFERENCE };

/**
 * @brief
 */
struct set_operation_fn {
  template <operation_type op_type>
  static std::unique_ptr<column> invoke(lists_column_view const&,
                                        lists_column_view const&,
                                        rmm::cuda_stream_view,
                                        rmm::mr::device_memory_resource*)
  {
    CUDF_UNREACHABLE("Base implementation of `set_operation_fn` should not be reached.");
  }
};

template <typename... Args>
auto dispatch_operation(operation_type op_type, Args&&... args)
{
  switch (op_type) {
    case operation_type::INTERSECTION:
      return set_operation_fn::invoke<operation_type::INTERSECTION>(std::forward<Args>(args)...);
    case operation_type::UNION:
      return set_operation_fn::invoke<operation_type::UNION>(std::forward<Args>(args)...);
    case operation_type::DIFFERENCE:
      return set_operation_fn::invoke<operation_type::DIFFERENCE>(std::forward<Args>(args)...);
  }

  CUDF_UNREACHABLE("Unsupported type.");
}

template <>
std::unique_ptr<column> set_operation_fn::invoke<operation_type::INTERSECTION>(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

template <>
std::unique_ptr<column> set_operation_fn::invoke<operation_type::UNION>(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

template <>
std::unique_ptr<column> set_operation_fn::invoke<operation_type::DIFFERENCE>(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

}  // namespace

}  // namespace detail

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::dispatch_operation(
    detail::operation_type::INTERSECTION, lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::dispatch_operation(
    detail::operation_type::UNION, lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::dispatch_operation(
    detail::operation_type::DIFFERENCE, lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
