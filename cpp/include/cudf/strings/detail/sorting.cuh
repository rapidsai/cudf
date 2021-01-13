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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Comparator for sorting strings column rows.
 */
struct sort_strings_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool lhs_null{d_column.is_null(lhs)};
      bool rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        if (!ascending) thrust::swap(lhs_null, rhs_null);
        return null_prec == cudf::null_order::BEFORE ? !rhs_null : !lhs_null;
      }
    }
    auto const lhs_str = d_column.element<string_view>(lhs);
    auto const rhs_str = d_column.element<string_view>(rhs);
    auto const cmp     = lhs_str.compare(rhs_str);
    return ascending ? (cmp < 0) : (cmp > 0);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  cudf::null_order null_prec;
};

/**
 * @brief Returns an indices column that is the sorted rows of the
 * input strings column.
 *
 * @param strings Strings instance for this operation.
 * @param sort_order Sort strings in ascending or descending order.
 * @param null_precedence Sort nulls to the beginning or the end of the new column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Indices of the sorted rows.
 */
template <bool stable = false>
std::unique_ptr<cudf::column> sorted_order(
  strings_column_view const strings,
  cudf::order sort_order              = cudf::order::ASCENDING,
  cudf::null_order null_precedence    = cudf::null_order::BEFORE,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), strings.size(), mask_state::UNALLOCATED, stream, mr);
  auto d_indices = sorted_indices->mutable_view();
  thrust::sequence(
    rmm::exec_policy(stream), d_indices.begin<size_type>(), d_indices.end<size_type>(), 0);

  sort_strings_comparator comparator{
    d_column, strings.has_nulls(), sort_order == cudf::order::ASCENDING, null_precedence};
  if (stable) {
    thrust::stable_sort(rmm::exec_policy(stream),
                        d_indices.begin<size_type>(),
                        d_indices.end<size_type>(),
                        comparator);
  } else {
    thrust::sort(rmm::exec_policy(stream),
                 d_indices.begin<size_type>(),
                 d_indices.end<size_type>(),
                 comparator);
  }
  return sorted_indices;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
