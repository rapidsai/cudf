/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "segmented_sort_impl.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common<sort_method::STABLE>(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common<sort_method::STABLE>(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, cudf::get_default_stream(), mr);
}

}  // namespace cudf
