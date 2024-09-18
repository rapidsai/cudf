/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

namespace cudf {

namespace {
struct string_view_to_pair {
  string_view null_placeholder;
  string_view_to_pair(string_view n) : null_placeholder(n) {}
  __device__ thrust::pair<char const*, size_type> operator()(string_view const& i)
  {
    return (i.data() == null_placeholder.data())
             ? thrust::pair<char const*, size_type>{nullptr, 0}
             : thrust::pair<char const*, size_type>{i.data(), i.size_bytes()};
  }
};

}  // namespace

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
  device_span<thrust::pair<char const*, size_type> const> strings,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return cudf::strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

std::unique_ptr<column> make_strings_column(device_span<string_view const> string_views,
                                            string_view null_placeholder,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto it_pair =
    thrust::make_transform_iterator(string_views.begin(), string_view_to_pair{null_placeholder});
  return cudf::strings::detail::make_strings_column(
    it_pair, it_pair + string_views.size(), stream, mr);
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            rmm::device_buffer&& chars_buffer,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask)
{
  CUDF_FUNC_RANGE();

  if (null_count > 0) { CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable."); }
  CUDF_EXPECTS(num_strings == offsets_column->size() - 1,
               "Invalid offsets column size for strings column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  std::move(chars_buffer),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            rmm::device_uvector<size_type>&& offsets,
                                            rmm::device_uvector<char>&& chars,
                                            rmm::device_buffer&& null_mask,
                                            size_type null_count)
{
  CUDF_FUNC_RANGE();

  if (num_strings == 0) { return make_empty_column(type_id::STRING); }

  auto const offsets_size = static_cast<size_type>(offsets.size());

  if (null_count > 0) CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable.");

  CUDF_EXPECTS(num_strings == offsets_size - 1, "Invalid offsets column size for strings column.");

  auto offsets_column = std::make_unique<column>(  //
    data_type{type_id::INT32},
    offsets_size,
    offsets.release(),
    rmm::device_buffer(),
    0);

  auto children = std::vector<std::unique_ptr<column>>();

  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  chars.release(),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
