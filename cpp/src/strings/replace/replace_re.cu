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

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the new string within the matched range of characters.
 *
 * The logic includes computing the size of each string and also writing the output.
 *
 * The stack is used to keep progress on evaluating the regex instructions on each string.
 * So the size of the stack is in proportion to the number of instructions in the given regex
 * pattern.
 *
 * There are three call types based on the number of regex instructions in the given pattern.
 * Small to medium instruction lengths can use the stack effectively though smaller executes faster.
 * Longer patterns require global memory. Shorter patterns are common in data cleaning.
 */
template <int stack_size>
struct replace_regex_fn {
  column_device_view const d_strings;
  reprog_device prog;
  string_view const d_repl;
  size_type const maxrepl;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();                  // number of characters in input string
    auto nbytes       = d_str.size_bytes();              // number of bytes in input string
    auto mxn          = maxrepl < 0 ? nchars : maxrepl;  // max possible replaces for this string
    auto in_ptr       = d_str.data();                    // input pointer (i)
    auto out_ptr      = d_chars ? d_chars + d_offsets[idx] : nullptr;  // output pointer (o)
    size_type lpos    = 0;
    int32_t begin     = 0;
    int32_t end       = static_cast<int32_t>(nchars);
    // copy input to output replacing strings as we go
    while (mxn-- > 0)  // maximum number of replaces
    {
      if (prog.is_empty() || prog.find<stack_size>(idx, d_str, begin, end) <= 0)
        break;                                        // no more matches
      auto spos = d_str.byte_offset(begin);           // get offset for these
      auto epos = d_str.byte_offset(end);             // character position values
      nbytes += d_repl.size_bytes() - (epos - spos);  // compute new size
      if (out_ptr)                                    // replace
      {                                               // i:bbbbsssseeee
        out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);  // o:bbbb
        out_ptr = copy_string(out_ptr, d_repl);                             // o:bbbbrrrrrr
                                                                            //  out_ptr ---^
        lpos = epos;                                                        // i:bbbbsssseeee
      }                                                                     //  in_ptr --^
      begin = end;
      end   = static_cast<int32_t>(nchars);
    }
    if (out_ptr)                                                  // copy the remainder
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);  // o:bbbbrrrrrreeee
    else
      d_offsets[idx] = static_cast<int32_t>(nbytes);
  }
};

}  // namespace

//
std::unique_ptr<column> replace_re(
  strings_column_view const& strings,
  std::string const& pattern,
  string_scalar const& replacement,
  std::optional<size_type> max_replace_count,
  regex_flags const flags,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  string_view d_repl(replacement.data(), replacement.size());

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // compile regex into device object
  auto prog =
    reprog_device::create(pattern, flags, get_character_flags_table(), strings_count, stream);
  auto d_prog            = *prog;
  auto const regex_insts = d_prog.insts_counts();

  // copy null mask
  auto null_mask        = cudf::detail::copy_bitmask(strings.parent(), stream, mr);
  auto const null_count = strings.null_count();
  auto const maxrepl    = max_replace_count.value_or(-1);

  // create child columns
  auto children = [&] {
    // Each invocation is predicated on the stack size which is dependent on the number of regex
    // instructions
    if (regex_insts <= RX_SMALL_INSTS) {
      replace_regex_fn<RX_STACK_SMALL> fn{d_strings, d_prog, d_repl, maxrepl};
      return make_strings_children(fn, strings_count, stream, mr);
    } else if (regex_insts <= RX_MEDIUM_INSTS) {
      replace_regex_fn<RX_STACK_MEDIUM> fn{d_strings, d_prog, d_repl, maxrepl};
      return make_strings_children(fn, strings_count, stream, mr);
    } else if (regex_insts <= RX_LARGE_INSTS) {
      replace_regex_fn<RX_STACK_LARGE> fn{d_strings, d_prog, d_repl, maxrepl};
      return make_strings_children(fn, strings_count, stream, mr);
    } else {
      replace_regex_fn<RX_STACK_ANY> fn{d_strings, d_prog, d_repl, maxrepl};
      return make_strings_children(fn, strings_count, stream, mr);
    }
  }();

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_re(strings_column_view const& strings,
                                   std::string const& pattern,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   regex_flags const flags,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_re(
    strings, pattern, replacement, max_replace_count, flags, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
