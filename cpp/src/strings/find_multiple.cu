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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> find_multiple(
  strings_column_view const& strings,
  strings_column_view const& targets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::INT32});
  auto targets_count = targets.size();
  CUDF_EXPECTS(targets_count > 0, "Must include at least one search target");
  CUDF_EXPECTS(!targets.has_nulls(), "Search targets cannot contain null strings");

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;

  // create output column
  auto total_count  = strings_count * targets_count;
  auto results      = make_numeric_column(data_type{type_id::INT32},
                                     total_count,
                                     rmm::device_buffer{0, stream, mr},
                                     0,
                                     stream,
                                     mr);  // no nulls
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<int32_t>();
  // fill output column with position values
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(total_count),
                    d_results,
                    [d_strings, d_targets, targets_count] __device__(size_type idx) {
                      size_type str_idx = idx / targets_count;
                      if (d_strings.is_null(str_idx)) return -1;
                      string_view d_str = d_strings.element<string_view>(str_idx);
                      string_view d_tgt = d_targets.element<string_view>(idx % targets_count);
                      return d_str.find(d_tgt);
                    });
  results->set_null_count(0);
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> find_multiple(strings_column_view const& strings,
                                      strings_column_view const& targets,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_multiple(strings, targets, mr);
}

}  // namespace strings
}  // namespace cudf
