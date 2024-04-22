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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/merge.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  using cudf::detail::side;
  if (row_order.is_empty()) { return make_empty_column(type_id::STRING); }
  auto const strings_count = static_cast<cudf::size_type>(row_order.size());

  auto const lhs_column = column_device_view::create(lhs.parent(), stream);
  auto const d_lhs      = *lhs_column;
  auto const rhs_column = column_device_view::create(rhs.parent(), stream);
  auto const d_rhs      = *rhs_column;

  auto const begin = row_order.begin();

  // build vector of strings
  rmm::device_uvector<string_index_pair> indices(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    indices.begin(),
                    [d_lhs, d_rhs, begin] __device__(size_type idx) {
                      auto const [s, index] = begin[idx];
                      if (s == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) {
                        return string_index_pair{nullptr, 0};
                      }
                      auto d_str = (s == side::LEFT) ? d_lhs.element<string_view>(index)
                                                     : d_rhs.element<string_view>(index);
                      return d_str.size_bytes() == 0
                               ? string_index_pair{"", 0}  // ensures empty != null
                               : string_index_pair{d_str.data(), d_str.size_bytes()};
                    });

  // convert vector into strings column
  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
