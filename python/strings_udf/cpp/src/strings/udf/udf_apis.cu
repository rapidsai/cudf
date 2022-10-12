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

#include <cudf/strings/udf/udf_apis.hpp>
#include <cudf/strings/udf/udf_string.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace udf {
namespace detail {
namespace {

struct free_udf_string_fn {
  cudf::strings::udf::udf_string* d_strings;
  __device__ void operator()(cudf::size_type idx) { d_strings[idx].clear(); }
};

void free_udf_string_array(void* d_buffer, std::size_t buffer_size, rmm::cuda_stream_view stream)
{
  auto const size =
    static_cast<cudf::size_type>(buffer_size / sizeof(cudf::strings::udf::udf_string));
  auto d_strings = reinterpret_cast<cudf::strings::udf::udf_string*>(d_buffer);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(0),
                     size,
                     free_udf_string_fn{d_strings});
}

struct udf_string_to_string_view_transform_fn {
  __device__ cudf::string_view operator()(cudf::strings::udf::udf_string const& dstr)
  {
    return cudf::string_view{dstr.data(), dstr.size_bytes()};
  }
};

}  // namespace

std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input,
                                                         rmm::cuda_stream_view stream)
{
  return std::make_unique<rmm::device_buffer>(
    std::move(cudf::strings::detail::create_string_vector_from_column(
                cudf::strings_column_view(input), stream)
                .release()));
}

std::unique_ptr<cudf::column> column_from_udf_string_array(void* d_buffer,
                                                           std::size_t buffer_size,
                                                           rmm::cuda_stream_view stream)
{
  auto const size =
    static_cast<cudf::size_type>(buffer_size / sizeof(cudf::strings::udf::udf_string));
  auto d_input = reinterpret_cast<cudf::strings::udf::udf_string*>(d_buffer);

  // create string_views of the udf_strings
  auto indices = rmm::device_uvector<cudf::string_view>(size, stream);
  thrust::transform(rmm::exec_policy(stream),
                    d_input,
                    d_input + size,
                    indices.data(),
                    udf_string_to_string_view_transform_fn{});

  auto results = cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), stream);

  // free the individual udf_string elements
  free_udf_string_array(d_buffer, buffer_size, stream);

  // return new column
  return results;
}

}  // namespace detail

std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input)
{
  return detail::to_string_view_array(input, rmm::cuda_stream_default);
}

std::unique_ptr<cudf::column> column_from_udf_string_array(void* d_buffer, std::size_t buffer_size)
{
  return detail::column_from_udf_string_array(d_buffer, buffer_size, rmm::cuda_stream_default);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
