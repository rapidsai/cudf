/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/pair.h>

namespace cudf {
namespace strings {
namespace detail {

using backref_type = thrust::pair<size_type, size_type>;

/**
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the at the backref position indicated in the replacement template.
 *
 * The logic includes computing the size of each string and also writing the output.
 */
template <typename Iterator>
struct backrefs_fn {
  column_device_view const d_strings;
  string_view const d_repl;  // string replacement template
  Iterator backrefs_begin;
  Iterator backrefs_end;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const in_ptr = d_str.data();
    auto const nchars = d_str.length();      // number of characters in input string
    auto nbytes       = d_str.size_bytes();  // number of bytes for the output string
    auto out_ptr      = d_chars ? (d_chars + d_offsets[idx]) : nullptr;
    size_type lpos    = 0;       // last byte position processed in d_str
    size_type begin   = 0;       // first character position matching regex
    size_type end     = nchars;  // last character position (exclusive)

    // copy input to output replacing strings as we go
    while (prog.find(prog_idx, d_str, begin, end) > 0)  // inits the begin/end vars
    {
      auto spos = d_str.byte_offset(begin);           // get offset for the
      auto epos = d_str.byte_offset(end);             // character position values;
      nbytes += d_repl.size_bytes() - (epos - spos);  // compute the output size

      // copy the string data before the matched section
      if (out_ptr) { out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos); }
      size_type lpos_template = 0;              // last end pos of replace template
      auto const repl_ptr     = d_repl.data();  // replace template pattern

      thrust::for_each(
        thrust::seq, backrefs_begin, backrefs_end, [&] __device__(backref_type backref) {
          if (out_ptr) {
            auto const copy_length = backref.second - lpos_template;
            out_ptr = copy_and_increment(out_ptr, repl_ptr + lpos_template, copy_length);
            lpos_template += copy_length;
          }
          // extract the specific group's string for this backref's index
          auto extracted = prog.extract(prog_idx, d_str, begin, end, backref.first - 1);
          if (!extracted || (extracted.value().second <= extracted.value().first)) {
            return;  // no value for this backref number; that is ok
          }
          auto spos_extract = d_str.byte_offset(extracted.value().first);   // convert
          auto epos_extract = d_str.byte_offset(extracted.value().second);  // to bytes
          nbytes += epos_extract - spos_extract;
          if (out_ptr) {
            out_ptr =
              copy_and_increment(out_ptr, in_ptr + spos_extract, (epos_extract - spos_extract));
          }
        });

      // copy remainder of template
      if (out_ptr && (lpos_template < d_repl.size_bytes())) {
        out_ptr = copy_and_increment(
          out_ptr, repl_ptr + lpos_template, d_repl.size_bytes() - lpos_template);
      }

      // setup to match the next section
      lpos  = epos;
      begin = end;
      end   = nchars;
    }

    // finally, copy remainder of input string
    if (out_ptr && (lpos < d_str.size_bytes())) {
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    } else if (!out_ptr) {
      d_offsets[idx] = static_cast<int32_t>(nbytes);
    }
  }
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
