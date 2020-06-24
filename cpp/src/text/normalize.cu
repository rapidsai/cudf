/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.cuh>

#include <nvtext/normalize.hpp>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/for_each.h>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Normalize spaces in a strings column.
 *
 * Repeated whitespace (code-point <= ' ') is replaced with a single space.
 * Also, whitespace is trimmed from the beginning and end of each string.
 *
 * This functor can be called to compute the output size in bytes
 * of each string and then called again to fill in the allocated buffer.
 */
struct normalize_spaces_fn {
  cudf::column_device_view const d_strings;  // strings to normalize
  int32_t const* d_offsets{};                // offsets into d_buffer
  char* d_buffer{};                          // output buffer for characters

  __device__ int32_t operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view single_space(" ", 1);
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    char* buffer     = d_offsets ? d_buffer + d_offsets[idx] : nullptr;
    char* optr       = buffer;  // running output pointer
    int32_t nbytes   = 0;       // holds the number of bytes per output string
    // create tokenizer for this string with whitespace delimiter (default)
    characters_tokenizer tokenizer(d_str);
    // this will retrieve tokens automatically skipping runs of whitespace
    while (tokenizer.next_token()) {
      auto token_pos = tokenizer.token_byte_positions();
      nbytes += token_pos.second - token_pos.first + 1;  // token size plus a single space
      if (optr) {
        cudf::string_view token(d_str.data() + token_pos.first, token_pos.second - token_pos.first);
        if (optr != buffer)  // prepend space unless we are at the beginning
          optr = cudf::strings::detail::copy_string(optr, single_space);
        // write token to output buffer
        optr = cudf::strings::detail::copy_string(optr, token);  // copy token to output
      }
    }
    return (nbytes > 0) ? nbytes - 1 : 0;  // remove trailing space
  }
};

}  // namespace

// details API
std::unique_ptr<cudf::column> normalize_spaces(
  cudf::strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  cudf::size_type strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create device column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // copy bitmask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);

  // create offsets by calculating size of each string for output
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    normalize_spaces_fn{d_strings});  // this does size-only calc
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build the chars column
  cudf::size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
  auto chars_column     = cudf::strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();

  // copy tokens to the chars buffer
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     normalize_spaces_fn{d_strings, d_offsets, d_chars});
  chars_column->set_null_count(0);  // reset null count for child column
  //
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   strings.null_count(),
                                   std::move(null_mask),
                                   stream,
                                   mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(strings, mr);
}

}  // namespace nvtext
