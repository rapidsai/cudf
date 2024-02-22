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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/find.h>

namespace cudf::io::json::detail {

// Extract the first character position in the string.
size_type find_first_delimiter(device_span<char const> d_data,
                               char const delimiter,
                               rmm::cuda_stream_view stream)
{
  auto const first_delimiter_position =
    thrust::find(rmm::exec_policy(stream), d_data.begin(), d_data.end(), delimiter);
  return first_delimiter_position != d_data.end() ? first_delimiter_position - d_data.begin() : -1;
}

}  // namespace cudf::io::json::detail
