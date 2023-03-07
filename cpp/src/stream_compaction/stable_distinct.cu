/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto const distinct_indices = get_distinct_indices(input.select(keys),
                                                     keep,
                                                     nulls_equal,
                                                     nans_equal,
                                                     stream,
                                                     rmm::mr::get_current_device_resource());

  // Markers to denote which rows to be copied to the output.
  auto const output_markers = [&] {
    auto markers = rmm::device_uvector<bool>(input.num_rows(), stream);
    thrust::uninitialized_fill(rmm::exec_policy(stream), markers.begin(), markers.end(), false);
    thrust::scatter(
      rmm::exec_policy(stream),
      thrust::constant_iterator<bool>(true, 0),
      thrust::constant_iterator<bool>(true, static_cast<size_type>(distinct_indices.size())),
      distinct_indices.begin(),
      markers.begin());
    return markers;
  }();

  return cudf::detail::copy_if(
    input,
    [output_markers = output_markers.begin()] __device__(auto const idx) {
      return *(output_markers + idx);
    },
    stream,
    mr);
}

}  // namespace cudf::detail
