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

#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the new string within the matched range of characters.
 */
struct replace_regex_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const maxrepl;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    auto nbytes       = d_str.size_bytes();              // number of bytes in input string
    auto mxn      = maxrepl < 0 ? nchars + 1 : maxrepl;  // max possible replaces for this string
    auto in_ptr   = d_str.data();                        // input pointer (i)
    auto out_ptr  = d_chars ? d_chars + d_offsets[idx]   // output pointer (o)
                            : nullptr;
    auto itr      = d_str.begin();
    auto last_pos = itr;

    // copy input to output replacing strings as we go
    while (mxn-- > 0 && itr.position() <= nchars && !prog.is_empty()) {
      auto const match = prog.find(prog_idx, d_str, itr);
      if (!match) { break; }  // no more matches

      auto const [start_pos, end_pos] = match_positions_to_bytes(*match, d_str, last_pos);
      nbytes += d_repl.size_bytes() - (end_pos - start_pos);  // add new size

      if (out_ptr) {                                                       // replace:
                                                                           // i:bbbbsssseeee
        out_ptr = copy_and_increment(out_ptr,                              //   ^
                                     in_ptr + last_pos.byte_offset(),      // o:bbbb
                                     start_pos - last_pos.byte_offset());  //       ^
        out_ptr = copy_string(out_ptr, d_repl);                            // o:bbbbrrrrrr
      }                                                                    //  out_ptr ---^
      last_pos += (match->second - last_pos.position());                   // i:bbbbsssseeee
                                                                           //  in_ptr --^

      itr = last_pos + (match->first == match->second);
    }

    if (out_ptr) {
      thrust::copy_n(thrust::seq,                                  // copy the remainder
                     in_ptr + last_pos.byte_offset(),              // o:bbbbrrrrrreeee
                     d_str.size_bytes() - last_pos.byte_offset(),  //             ^   ^
                     out_ptr);
    } else {
      d_sizes[idx] = nbytes;
    }
  }
};

}  // namespace

//
std::unique_ptr<column> replace_re(strings_column_view const& input,
                                   regex_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  string_view d_repl(replacement.data(), replacement.size());

  // create device object from regex_program
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  auto const maxrepl = max_replace_count.value_or(-1);

  auto const d_strings = column_device_view::create(input.parent(), stream);

  auto [offsets_column, chars] = make_strings_children(
    replace_regex_fn{*d_strings, d_repl, maxrepl}, *d_prog, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_re(strings_column_view const& strings,
                                   regex_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_re(strings, prog, replacement, max_replace_count, stream, mr);
}

}  // namespace strings
}  // namespace cudf
