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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/udf/column_functions.hpp>
#include <cudf/strings/udf/dstring.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace udf {

namespace {

struct free_dstring_fn {
  cudf::strings::udf::dstring* d_strings;
  __device__ void operator()(cudf::size_type idx) { d_strings[idx].clear(); }
};

struct dstring_to_string_view_transform_fn {
  __device__ cudf::string_view operator()(cudf::strings::udf::dstring const& d_str)
  {
    return cudf::string_view{d_str.data(), d_str.size_bytes()};
  }
};

}  // namespace

namespace detail {

std::unique_ptr<rmm::device_buffer> create_dstring_array(cudf::size_type size,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  auto const output_vector_size = size * sizeof(cudf::strings::udf::dstring);
  auto result = std::make_unique<rmm::device_buffer>(output_vector_size, stream, mr);
  cudaMemset(result->data(), 0, output_vector_size);
  return result;
}

void free_dstring_array(device_span<dstring> input, rmm::cuda_stream_view stream)
{
  auto const size = static_cast<cudf::size_type>(input.size());
  auto d_strings  = input.data();
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator(0), size, free_dstring_fn{d_strings});
}

std::unique_ptr<cudf::column> make_strings_column(device_span<dstring const> input,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto const size = static_cast<cudf::size_type>(input.size());
  auto d_input    = input.data();

  // create string_views of the dstrings
  auto indices = rmm::device_uvector<cudf::string_view>(size, stream);
  thrust::transform(rmm::exec_policy(stream),
                    d_input,
                    d_input + size,
                    indices.data(),
                    dstring_to_string_view_transform_fn{});

  return cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), stream);
}

}  // namespace detail

rmm::device_uvector<string_view> create_string_view_array(cudf::strings_column_view const input,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::strings::detail::create_string_vector_from_column(cudf::strings_column_view(input),
                                                                 rmm::cuda_stream_default);
}

std::unique_ptr<rmm::device_buffer> create_dstring_array(size_type size,
                                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::create_dstring_array(size, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> make_strings_column(device_span<dstring const> input,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::make_strings_column(input, rmm::cuda_stream_default, mr);
}

void free_dstring_array(device_span<dstring> input)
{
  CUDF_FUNC_RANGE();
  return detail::free_dstring_array(input, rmm::cuda_stream_default);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
