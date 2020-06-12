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
#include <cudf/strings/string_view.cuh>
#include <strings/regex/regex.cuh>
#include <strings/utilities.cuh>

namespace cudf {
namespace strings {
namespace detail {

using backref_type = thrust::pair<size_type, size_type>;

/**
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the at the backref position indicated in the replacement template.
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
 *
 */
template <size_t stack_size>
struct backrefs_fn {
  column_device_view const d_strings;
  reprog_device prog;
  string_view const d_repl;  // string replacement template
  rmm::device_vector<backref_type>::iterator backrefs_begin;
  rmm::device_vector<backref_type>::iterator backrefs_end;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    u_char data1[stack_size];
    u_char data2[stack_size];
    prog.set_stack_mem(data1, data2);
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();      // number of characters in input string
    auto nbytes       = d_str.size_bytes();  // number of bytes in input string
    auto in_ptr       = d_str.data();
    auto out_ptr      = d_chars ? (d_chars + d_offsets[idx]) : nullptr;
    size_type lpos    = 0;       // last byte position processed in d_str
    size_type begin   = 0;       // first character position matching regex
    size_type end     = nchars;  // last character position (exclusive)
    // copy input to output replacing strings as we go
    while (prog.find(idx, d_str, begin, end) > 0)  // inits the begin/end vars
    {
      auto spos = d_str.byte_offset(begin);           // get offset for these
      auto epos = d_str.byte_offset(end);             // character position values
      nbytes += d_repl.size_bytes() - (epos - spos);  // compute new size
      if (out_ptr) out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
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
          size_type spos_extract = begin;  // these are modified
          size_type epos_extract = end;    // by extract()
          if ((prog.extract(idx, d_str, spos_extract, epos_extract, backref.first - 1) <= 0) ||
              (epos_extract <= spos_extract))
            return;  // no value for this backref number; that is ok
          spos_extract = d_str.byte_offset(spos_extract);  // convert
          epos_extract = d_str.byte_offset(epos_extract);  // to bytes
          nbytes += epos_extract - spos_extract;
          if (out_ptr)
            out_ptr =
              copy_and_increment(out_ptr, in_ptr + spos_extract, (epos_extract - spos_extract));
        });
      if (out_ptr && (lpos_template < d_repl.size_bytes()))  // copy remainder of template
        out_ptr = copy_and_increment(
          out_ptr, repl_ptr + lpos_template, d_repl.size_bytes() - lpos_template);
      lpos  = epos;
      begin = end;
      end   = nchars;
    }
    if (out_ptr && (lpos < d_str.size_bytes()))  // copy remainder of input string
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    else if (!out_ptr)
      d_offsets[idx] = nbytes;
  }
};

using children_pair = std::pair<std::unique_ptr<column>, std::unique_ptr<column>>;

children_pair replace_with_backrefs_medium(column_device_view const& d_strings,
                                           reprog_device& d_prog,
                                           string_view const& d_repl_template,
                                           rmm::device_vector<backref_type>& backrefs,
                                           size_type null_count,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream);

children_pair replace_with_backrefs_large(column_device_view const& d_strings,
                                          reprog_device& d_prog,
                                          string_view const& d_repl_template,
                                          rmm::device_vector<backref_type>& backrefs,
                                          size_type null_count,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
