/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/strip.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Strip characters from the beginning and/or end of a string
 *
 * This functor strips the beginning and/or end of each string
 * of any characters found in d_to_strip or whitespace if
 * d_to_strip is empty.
 *
 */
struct strip_transform_fn {
  column_device_view const d_strings;
  side_type const side;  // right, left, or both
  string_view const d_to_strip;

  __device__ string_index_pair operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str      = d_strings.element<string_view>(idx);
    auto const d_stripped = strip(d_str, d_to_strip, side);
    return string_index_pair{d_stripped.data(), d_stripped.size_bytes()};
  }
};

}  // namespace

std::unique_ptr<column> strip(strings_column_view const& input,
                              side_type side,
                              string_scalar const& to_strip,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(to_strip.is_valid(stream), "Parameter to_strip must be valid");
  string_view const d_to_strip(to_strip.data(), to_strip.size());

  auto const d_column = column_device_view::create(input.parent(), stream);

  auto result = rmm::device_uvector<string_index_pair>(input.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(input.size()),
                    result.begin(),
                    strip_transform_fn{*d_column, side, d_to_strip});

  return make_strings_column(result.begin(), result.end(), stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> strip(strings_column_view const& input,
                              side_type side,
                              string_scalar const& to_strip,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::strip(input, side, to_strip, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
