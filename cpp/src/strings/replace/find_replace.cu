/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
struct find_replace_fn {
  column_device_view d_input;
  column_device_view d_values;
  column_device_view d_replacements;

  __device__ string_index_pair get_replacement(size_type idx)
  {
    if (d_replacements.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str = d_replacements.element<string_view>(idx);
    return string_index_pair{d_str.data(), d_str.size_bytes()};
  }

  __device__ string_index_pair operator()(size_type idx)
  {
    if (d_input.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str = d_input.element<string_view>(idx);
    // find d_str in d_values
    // if found return corresponding replacement
    // if not found, return d_str
    auto const begin = thrust::counting_iterator<size_type>(0);
    auto const end   = thrust::counting_iterator<size_type>(d_values.size());
    auto const itr =
      thrust::find_if(thrust::seq, begin, end, [d_values = d_values, d_str](size_type i) -> bool {
        return d_str == d_values.element<string_view>(i);
      });
    return itr == end ? string_index_pair{d_str.data(), d_str.size_bytes()} : get_replacement(*itr);
  }
};

}  // namespace

std::unique_ptr<cudf::column> find_and_replace_all(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& values_to_replace,
  cudf::strings_column_view const& replacement_values,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto d_input             = cudf::column_device_view::create(input.parent(), stream);
  auto d_values_to_replace = cudf::column_device_view::create(values_to_replace.parent(), stream);
  auto d_replacements      = cudf::column_device_view::create(replacement_values.parent(), stream);

  auto indices = rmm::device_uvector<string_index_pair>(input.size(), stream);

  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(input.size()),
                    indices.begin(),
                    find_replace_fn{*d_input, *d_values_to_replace, *d_replacements});

  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
