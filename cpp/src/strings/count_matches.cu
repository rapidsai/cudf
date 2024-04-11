/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "strings/count_matches.hpp"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Kernel counts the total matches for the given regex in each string.
 */
struct count_fn {
  column_device_view const d_strings;

  __device__ int32_t operator()(size_type const idx,
                                reprog_device const prog,
                                int32_t const thread_idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    int32_t count     = 0;

    auto itr = d_str.begin();
    while (itr.position() <= nchars) {
      auto result = prog.find(thread_idx, d_str, itr);
      if (!result) { break; }
      ++count;
      // increment the iterator is faster than creating a new one
      // +1 if the match was on a virtual position (e.g. word boundary)
      itr += (result->second - itr.position()) + (result->first == result->second);
    }
    return count;
  }
};

}  // namespace

std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      reprog_device& d_prog,
                                      size_type output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  assert(output_size >= d_strings.size() and "Unexpected output size");

  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, output_size, mask_state::UNALLOCATED, stream, mr);

  if (d_strings.size() == 0) return results;

  auto d_results = results->mutable_view().data<int32_t>();

  launch_transform_kernel(count_fn{d_strings}, d_prog, d_results, d_strings.size(), stream);

  return results;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
