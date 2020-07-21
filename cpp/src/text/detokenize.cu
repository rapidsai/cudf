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
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/tokenize.hpp>
#include <strings/utilities.cuh>

#include <thrust/transform_scan.h>
#include <rmm/device_uvector.hpp>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Generate strings from tokens.
 *
 * Each string is created by appending all the tokens assigned to
 * the same row. The `d_separator` is appended between each token.
 */
struct detokenizer_fn {
  cudf::column_device_view const d_strings;  // these are the tokens
  int32_t const* d_row_map;                  // indices sorted by output row
  cudf::size_type const* d_token_offsets;    // to each input token array
  cudf::string_view const d_separator;       // append after each token
  int32_t const* d_offsets{};                // offsets to output buffer d_chars
  char* d_chars{};                           // output buffer for characters

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    auto const offset      = d_token_offsets[idx];
    auto d_tokens          = d_row_map + offset;
    auto const token_count = d_token_offsets[idx + 1] - offset;
    auto out_ptr           = d_chars ? d_chars + d_offsets[idx] : nullptr;
    cudf::size_type nbytes = 0;
    for (cudf::size_type jdx = 0; jdx < token_count; ++jdx) {
      auto const str_index = d_tokens[jdx];
      if (d_strings.is_null(str_index)) continue;
      auto const d_str = d_strings.element<cudf::string_view>(str_index);
      if (out_ptr) {
        out_ptr = cudf::strings::detail::copy_string(out_ptr, d_str);
        if (jdx + 1 < token_count)
          out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator);
      } else {
        nbytes += d_str.size_bytes();
        nbytes += d_separator.size_bytes();
      }
    }
    return (nbytes > 0) ? (nbytes - d_separator.size_bytes()) : 0;
  }
};

}  // namespace

std::unique_ptr<cudf::column> detokenize(cudf::strings_column_view const& strings,
                                         cudf::column_view const& row_indices,
                                         cudf::string_scalar const& separator,
                                         cudaStream_t stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be valid");
  CUDF_EXPECTS(row_indices.size() == strings.size(),
               "Parameter row_indices must be the same size as the input column");
  CUDF_EXPECTS(row_indices.has_nulls() == false, "Parameter row_indices must not have nulls");

  auto tokens_count = strings.size();
  if (tokens_count == 0)  // if no strings, return an empty column
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto execpol        = rmm::exec_policy(stream);
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  auto d_rows      = row_indices.data<int32_t>();  // TODO: this is type-dispatchable
  auto sorted_rows = cudf::stable_sorted_order(cudf::table_view({row_indices}));
  auto d_row_map   = sorted_rows->view().data<int32_t>();

  // the number of output strings is the last entry
  int32_t last_entry =
    cudf::detail::get_value<int32_t>(sorted_rows->view(), tokens_count - 1, stream);
  auto output_count = cudf::detail::get_value<int32_t>(row_indices, last_entry, stream) + 1;

  // create offsets for the tokens for each output string
  rmm::device_uvector<cudf::size_type> token_offsets(output_count + 1, stream);
  thrust::copy_if(execpol->on(stream),
                  thrust::make_counting_iterator<cudf::size_type>(0),
                  thrust::make_counting_iterator<cudf::size_type>(tokens_count),
                  token_offsets.begin(),
                  [output_count, d_rows, d_row_map] __device__(cudf::size_type idx) {
                    return (idx == 0) || (d_rows[d_row_map[idx]] != d_rows[d_row_map[idx - 1]]);
                  });
  token_offsets.set_element(output_count, tokens_count, stream);

  // create output strings offsets by calculating size of each output string
  cudf::string_view const d_separator(separator.data(), separator.size());
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0),
    detokenizer_fn{d_strings, d_row_map, token_offsets.data(), d_separator});
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + output_count, mr, stream);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build the chars column - append each source token to the appropriate output string
  cudf::size_type const total_bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), output_count, stream);
  auto chars_column =
    cudf::strings::detail::create_chars_child_column(output_count, 0, total_bytes, mr, stream);
  char* const d_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    output_count,
    detokenizer_fn{d_strings, d_row_map, token_offsets.data(), d_separator, d_offsets, d_chars});
  chars_column->set_null_count(0);

  // make the output strings column from the offsets and chars column
  return cudf::make_strings_column(output_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   0,
                                   rmm::device_buffer{0, stream, mr},
                                   stream,
                                   mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> detokenize(cudf::strings_column_view const& strings,
                                         cudf::column_view const& row_indices,
                                         cudf::string_scalar const& separator,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::detokenize(strings, row_indices, separator, 0, mr);
}

}  // namespace nvtext
