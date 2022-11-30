/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <strings/regex/regex_program_impl.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/pair.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
namespace {
// this is a [begin,end) pair of character positions when a substring is matched
using found_range = thrust::pair<size_type, size_type>;

/**
 * @brief This functor handles replacing strings by applying the compiled regex patterns
 * and inserting the corresponding new string within the matched range of characters.
 */
struct replace_multi_regex_fn {
  column_device_view const d_strings;
  device_span<reprog_device const> progs;  // array of regex progs
  found_range* d_found_ranges;             // working array matched (begin,end) values
  column_device_view const d_repls;        // replacement strings
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    auto const number_of_patterns = static_cast<size_type>(progs.size());

    auto const d_str      = d_strings.element<string_view>(idx);
    auto const nchars     = d_str.length();      // number of characters in input string
    auto nbytes           = d_str.size_bytes();  // number of bytes in input string
    auto in_ptr           = d_str.data();        // input pointer
    auto out_ptr          = d_chars ? d_chars + d_offsets[idx] : nullptr;
    found_range* d_ranges = d_found_ranges + (idx * number_of_patterns);
    size_type lpos        = 0;
    size_type ch_pos      = 0;
    // initialize the working ranges memory to -1's
    thrust::fill(thrust::seq, d_ranges, d_ranges + number_of_patterns, found_range{-1, 1});
    // process string one character at a time
    while (ch_pos < nchars) {
      // this minimizes the regex-find calls by only calling it for stale patterns
      // -- those that have not previously matched up to this point (ch_pos)
      for (size_type ptn_idx = 0; ptn_idx < number_of_patterns; ++ptn_idx) {
        if (d_ranges[ptn_idx].first >= ch_pos)  // previously matched here
          continue;                             // or later in the string
        reprog_device prog = progs[ptn_idx];

        auto begin = ch_pos;
        auto end   = nchars;
        if (!prog.is_empty() && prog.find(idx, d_str, begin, end) > 0)
          d_ranges[ptn_idx] = found_range{begin, end};  // found a match
        else
          d_ranges[ptn_idx] = found_range{nchars, nchars};  // this pattern is done
      }
      // all the ranges have been updated from each regex match;
      // look for any that match at this character position (ch_pos)
      auto itr =
        thrust::find_if(thrust::seq, d_ranges, d_ranges + number_of_patterns, [ch_pos](auto range) {
          return range.first == ch_pos;
        });
      if (itr != d_ranges + number_of_patterns) {
        // match found, compute and replace the string in the output
        size_type ptn_idx  = static_cast<size_type>(itr - d_ranges);
        size_type begin    = d_ranges[ptn_idx].first;
        size_type end      = d_ranges[ptn_idx].second;
        string_view d_repl = d_repls.size() > 1 ? d_repls.element<string_view>(ptn_idx)
                                                : d_repls.element<string_view>(0);
        auto spos          = d_str.byte_offset(begin);
        auto epos          = d_str.byte_offset(end);
        nbytes += d_repl.size_bytes() - (epos - spos);
        if (out_ptr) {  // copy unmodified content plus new replacement string
          out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
          out_ptr = copy_string(out_ptr, d_repl);
          lpos    = epos;
        }
        ch_pos = end - 1;
      }
      ++ch_pos;
    }
    if (out_ptr)  // copy the remainder
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    else
      d_offsets[idx] = static_cast<int32_t>(nbytes);
  }
};

}  // namespace

std::unique_ptr<column> replace_re(strings_column_view const& input,
                                   std::vector<std::string> const& patterns,
                                   strings_column_view const& replacements,
                                   regex_flags const flags,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  if (patterns.empty()) {  // if no patterns; just return a copy
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  CUDF_EXPECTS(!replacements.has_nulls(), "Parameter replacements must not have any nulls");

  // compile regexes into device objects
  auto h_progs = std::vector<std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>>(
    patterns.size());
  std::transform(
    patterns.begin(), patterns.end(), h_progs.begin(), [flags, stream](auto const& ptn) {
      auto h_prog = regex_program::create(ptn, flags, capture_groups::NON_CAPTURE);
      return regex_device_builder::create_prog_device(*h_prog, stream);
    });

  // get the longest regex for the dispatcher
  auto const max_prog =
    std::max_element(h_progs.begin(), h_progs.end(), [](auto const& lhs, auto const& rhs) {
      return lhs->insts_counts() < rhs->insts_counts();
    });

  auto d_max_prog        = **max_prog;
  auto const buffer_size = d_max_prog.working_memory_size(input.size());
  auto d_buffer          = rmm::device_buffer(buffer_size, stream);

  // copy all the reprog_device instances to a device memory array
  std::vector<reprog_device> progs;
  std::transform(h_progs.begin(),
                 h_progs.end(),
                 std::back_inserter(progs),
                 [d_buffer = d_buffer.data(), size = input.size()](auto& prog) {
                   prog->set_working_memory(d_buffer, size);
                   return *prog;
                 });
  auto d_progs = cudf::detail::make_device_uvector_async(progs, stream);

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_repls   = column_device_view::create(replacements.parent(), stream);

  auto found_ranges = rmm::device_uvector<found_range>(d_progs.size() * input.size(), stream);

  auto children = make_strings_children(
    replace_multi_regex_fn{*d_strings, d_progs, found_ranges.data(), *d_repls},
    input.size(),
    stream,
    mr);

  return make_strings_column(input.size(),
                             std::move(children.first),
                             std::move(children.second),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_re(strings_column_view const& strings,
                                   std::vector<std::string> const& patterns,
                                   strings_column_view const& replacements,
                                   regex_flags const flags,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_re(strings, patterns, replacements, flags, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
