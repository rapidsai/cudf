/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "regex.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/regex/config.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::strings {
namespace detail {

std::pair<std::size_t, size_type> compute_regex_state_memory(strings_column_view const& input,
                                                             std::string const& pattern,
                                                             regex_flags const flags,
                                                             rmm::cuda_stream_view stream)
{
  auto d_prog = reprog_device::create(pattern, flags, stream);
  return d_prog->compute_strided_working_memory(input.size());
}

}  // namespace detail

std::pair<std::size_t, size_type> compute_regex_state_memory(strings_column_view const& input,
                                                             std::string const& pattern,
                                                             regex_flags const flags)
{
  CUDF_FUNC_RANGE();
  return detail::compute_regex_state_memory(input, pattern, flags, rmm::cuda_stream_default);
}

}  // namespace cudf::strings
