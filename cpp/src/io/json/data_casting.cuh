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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf::io::json::experimental {

template <typename str_tuple_it>
rmm::device_uvector<thrust::pair<const char*, size_type>> coalesce_input(
  str_tuple_it str_tuples, size_type col_size, rmm::cuda_stream_view stream)
{
  auto result = rmm::device_uvector<thrust::pair<const char*, size_type>>(col_size, stream);
  thrust::copy_n(rmm::exec_policy(stream), str_tuples, col_size, result.begin());
  return result;
}

template <typename str_tuple_it>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   rmm::cuda_stream_view stream)
{
  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    auto const strings_span = coalesce_input(str_tuples, col_size, stream);
    return make_strings_column(strings_span, stream);
  } else {
    CUDF_FAIL("Type conversion not implemented");
    // full version: use existing code (`ConvertFunctor`) to convert values
  }
}

}  // namespace cudf::io::json::experimental
