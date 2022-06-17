/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "data_casting.cuh"

namespace cudf::io::json::experimental {

template <typename str_spans_it_it, typename col_size_it>
std::vector<std::unique_ptr<column>> parse_data(str_spans_it_it cols_str_spans,
                                                col_size_it cols_size,
                                                host_span<data_type const> cols_type,
                                                rmm::cuda_stream_view stream)
{
  // first version: make_strings_column from the string spans
  // full version: use existing code (`ConvertFunctor`) to convert values
  return {};
}

template std::vector<std::unique_ptr<column>> parse_data<string_view**, size_type*>(
  string_view** cols_str_spans,
  size_type* cols_size,
  host_span<data_type const> cols_type,
  rmm::cuda_stream_view stream);
}  // namespace cudf::io::json::experimental
