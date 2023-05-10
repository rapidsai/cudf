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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Reverse individual characters in each string
 */
struct reverse_characters_fn {
  column_device_view const d_strings;
  offset_type const* d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str = d_strings.element<string_view>(idx);
    // pointer to the end of the output string
    auto d_output = d_chars + d_offsets[idx] + d_str.size_bytes();
    for (auto const chr : d_str) {          // iterate through each character;
      d_output -= bytes_in_char_utf8(chr);  // position output;
      from_char_utf8(chr, d_output);        // place character into output
    }
  }
};

}  // namespace

std::unique_ptr<column> reverse(strings_column_view const& input,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }

  // copy the column; replace data in the chars column
  auto result = std::make_unique<column>(input.parent(), stream, mr);
  auto const d_offsets =
    result->view().child(strings_column_view::offsets_column_index).data<offset_type>();
  auto d_chars = result->mutable_view().child(strings_column_view::chars_column_index).data<char>();

  auto const d_column = column_device_view::create(input.parent(), stream);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<size_type>(0),
                     input.size(),
                     reverse_characters_fn{*d_column, d_offsets, d_chars});

  return result;
}

}  // namespace detail

std::unique_ptr<column> reverse(strings_column_view const& input,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(input, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
