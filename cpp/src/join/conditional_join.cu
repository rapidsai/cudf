/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <join/conditional_join.cuh>
#include <join/join_common_utils.hpp>

#include <cudf/ast/expressions.hpp>
//#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_join(table_view left,
                 table_view right,
                 ast::expression binary_predicate,
                 null_equality compare_nulls,
                 join_kind JoinKind,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return get_conditional_join_indices(
    left, right, JoinKind, binary_predicate, compare_nulls, stream, mr);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view left,
                       table_view right,
                       ast::expression binary_predicate,
                       null_equality compare_nulls,
                       rmm::mr::device_memory_resource* mr)
{
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  compare_nulls,
                                  detail::join_kind::INNER_JOIN,
                                  rmm::cuda_stream_default,
                                  mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view left,
                      table_view right,
                      ast::expression binary_predicate,
                      null_equality compare_nulls,
                      rmm::mr::device_memory_resource* mr)
{
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  compare_nulls,
                                  detail::join_kind::LEFT_JOIN,
                                  rmm::cuda_stream_default,
                                  mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join(table_view left,
                      table_view right,
                      ast::expression binary_predicate,
                      null_equality compare_nulls,
                      rmm::mr::device_memory_resource* mr)
{
  return detail::conditional_join(left,
                                  right,
                                  binary_predicate,
                                  compare_nulls,
                                  detail::join_kind::FULL_JOIN,
                                  rmm::cuda_stream_default,
                                  mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view left,
  table_view right,
  ast::expression binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  return std::move(detail::conditional_join(left,
                                            right,
                                            binary_predicate,
                                            compare_nulls,
                                            detail::join_kind::LEFT_SEMI_JOIN,
                                            rmm::cuda_stream_default,
                                            mr)
                     .first);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view left,
  table_view right,
  ast::expression binary_predicate,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  return std::move(detail::conditional_join(left,
                                            right,
                                            binary_predicate,
                                            compare_nulls,
                                            detail::join_kind::LEFT_ANTI_JOIN,
                                            rmm::cuda_stream_default,
                                            mr)
                     .first);
}

}  // namespace cudf
