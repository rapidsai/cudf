/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <strings/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {

namespace {
struct string_view_to_pair {
  string_view null_placeholder;
  string_view_to_pair(string_view n) : null_placeholder(n) {}
  __device__ thrust::pair<const char*, size_type> operator()(const string_view& i)
  {
    return (i.data() == null_placeholder.data())
             ? thrust::pair<const char*, size_type>{nullptr, 0}
             : thrust::pair<const char*, size_type>{i.data(), i.size_bytes()};
  }
};

}  // namespace

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
  device_span<thrust::pair<const char*, size_type> const> strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  return cudf::strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

std::unique_ptr<column> make_strings_column(
  device_span<char> chars,
  device_span<size_type> offsets,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_FUNC_RANGE();

  return cudf::strings::detail::make_strings_column(chars.begin(),
                                                    chars.end(),
                                                    offsets.begin(),
                                                    offsets.end(),
                                                    null_count,
                                                    std::move(null_mask),
                                                    stream,
                                                    mr);
}

std::unique_ptr<column> make_strings_column(device_span<string_view const> string_views,
                                            string_view null_placeholder,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  auto it_pair =
    thrust::make_transform_iterator(string_views.begin(), string_view_to_pair{null_placeholder});
  return cudf::strings::detail::make_strings_column(
    it_pair, it_pair + string_views.size(), stream, mr);
}

// Create a strings-type column from device vector of chars and vector of offsets.
std::unique_ptr<column> make_strings_column(cudf::device_span<char const> strings,
                                            cudf::device_span<size_type const> offsets,
                                            cudf::device_span<bitmask_type const> valid_mask,
                                            size_type null_count,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  // build null bitmask
  rmm::device_buffer null_mask{
    valid_mask.data(), valid_mask.size() * sizeof(bitmask_type), stream, mr};

  return cudf::strings::detail::make_strings_column(strings.begin(),
                                                    strings.end(),
                                                    offsets.begin(),
                                                    offsets.end(),
                                                    null_count,
                                                    std::move(null_mask),
                                                    stream,
                                                    mr);
}

//
std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            std::unique_ptr<column> chars_column,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (null_count > 0) CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable.");
  CUDF_EXPECTS(num_strings == offsets_column->size() - 1,
               "Invalid offsets column size for strings column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");
  CUDF_EXPECTS(chars_column->null_count() == 0, "Chars column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(chars_column));
  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  rmm::device_buffer{0, stream, mr},
                                  null_mask,
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
