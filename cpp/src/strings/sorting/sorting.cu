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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/strings/sorting.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace strings {
namespace detail {
// return sorted version of the given strings column
std::unique_ptr<cudf::column> sort(strings_column_view strings,
                                   sort_type stype,
                                   cudf::order order,
                                   cudf::null_order null_order,
                                   cudaStream_t stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto execpol        = rmm::exec_policy(stream);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // sort the indices of the strings
  size_type num_strings = strings.size();
  rmm::device_vector<size_type> indices(num_strings);
  thrust::sequence(execpol->on(stream), indices.begin(), indices.end());
  thrust::sort(execpol->on(stream),
               indices.begin(),
               indices.end(),
               [d_column, stype, order, null_order] __device__(size_type lhs, size_type rhs) {
                 bool lhs_null{d_column.is_null(lhs)};
                 bool rhs_null{d_column.is_null(rhs)};
                 if (lhs_null || rhs_null)
                   return (null_order == cudf::null_order::BEFORE ? !rhs_null : !lhs_null);
                 string_view lhs_str = d_column.element<string_view>(lhs);
                 string_view rhs_str = d_column.element<string_view>(rhs);
                 int cmp             = 0;
                 if (stype & sort_type::length) cmp = lhs_str.length() - rhs_str.length();
                 if (stype & sort_type::name) cmp = lhs_str.compare(rhs_str);
                 return (order == cudf::order::ASCENDING ? (cmp < 0) : (cmp > 0));
               });

  // create a column_view as a wrapper of these indices
  column_view indices_view(
    data_type{type_id::INT32}, num_strings, indices.data().get(), nullptr, 0);
  // now build a new strings column from the indices
  auto table_sorted = cudf::detail::gather(table_view{{strings.parent()}},
                                           indices_view,
                                           cudf::detail::out_of_bounds_policy::NULLIFY,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           mr,
                                           stream)
                        ->release();
  return std::move(table_sorted.front());
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
