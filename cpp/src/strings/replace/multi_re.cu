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
// this is a [begin,end) pair of character positions when a substring is matched
using found_range = thrust::pair<size_type, size_type>;

/**
 * @brief This functor handles replacing strings by applying the compiled regex patterns
 * and inserting the corresponding new string within the matched range of characters.
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
struct replace_multi_regex_fn {
  column_device_view const d_strings;
  reprog_device* progs;  // array of regex progs
  size_type number_of_patterns;
  found_range* d_found_ranges;       // working array matched (begin,end) values
  column_device_view const d_repls;  // replacment strings
  int32_t* d_offsets{};              // these are null when
  char* d_chars{};                   // only computing size

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
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

        auto begin = static_cast<int32_t>(ch_pos);
        auto end   = static_cast<int32_t>(nchars);
        if (!prog.is_empty() && prog.find<stack_size>(idx, d_str, begin, end) > 0)
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

std::unique_ptr<column> replace_re(
  strings_column_view const& strings,
  std::vector<std::string> const& patterns,
  strings_column_view const& repls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});
  if (patterns.empty())  // no patterns; just return a copy
    return std::make_unique<column>(strings.parent(), stream, mr);

  CUDF_EXPECTS(!repls.has_nulls(), "Parameter repls must not have any nulls");

  auto d_strings = column_device_view::create(strings.parent(), stream);
  auto d_repls   = column_device_view::create(repls.parent(), stream);
  auto d_flags   = get_character_flags_table();

  // compile regexes into device objects
  size_type regex_insts = 0;
  std::vector<std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>> h_progs;
  thrust::host_vector<reprog_device> progs;
  for (auto itr = patterns.begin(); itr != patterns.end(); ++itr) {
    auto prog   = reprog_device::create(*itr, d_flags, strings_count, stream);
    regex_insts = std::max(regex_insts, prog->insts_counts());
    progs.push_back(*prog);
    h_progs.emplace_back(std::move(prog));
  }

  // copy all the reprog_device instances to a device memory array
  rmm::device_buffer progs_buffer{sizeof(reprog_device) * progs.size(), stream};
  CUDA_TRY(cudaMemcpyAsync(progs_buffer.data(),
                           progs.data(),
                           progs.size() * sizeof(reprog_device),
                           cudaMemcpyHostToDevice,
                           stream.value()));
  reprog_device* d_progs = reinterpret_cast<reprog_device*>(progs_buffer.data());

  // create working buffer for ranges pairs
  rmm::device_uvector<found_range> found_ranges(patterns.size() * strings_count, stream);
  auto d_found_ranges = found_ranges.data();

  // create child columns
  auto children = [&] {
    // Each invocation is predicated on the stack size which is dependent on the number of regex
    // instructions
    if (regex_insts <= RX_SMALL_INSTS)
      return make_strings_children(
        replace_multi_regex_fn<RX_STACK_SMALL>{
          *d_strings, d_progs, static_cast<size_type>(progs.size()), d_found_ranges, *d_repls},
        strings_count,
        stream,
        mr);
    else if (regex_insts <= RX_MEDIUM_INSTS)
      return make_strings_children(
        replace_multi_regex_fn<RX_STACK_MEDIUM>{
          *d_strings, d_progs, static_cast<size_type>(progs.size()), d_found_ranges, *d_repls},
        strings_count,
        stream,
        mr);
    else if (regex_insts <= RX_LARGE_INSTS)
      return make_strings_children(
        replace_multi_regex_fn<RX_STACK_LARGE>{
          *d_strings, d_progs, static_cast<size_type>(progs.size()), d_found_ranges, *d_repls},
        strings_count,
        stream,
        mr);
    else
      return make_strings_children(
        replace_multi_regex_fn<RX_STACK_ANY>{
          *d_strings, d_progs, static_cast<size_type>(progs.size()), d_found_ranges, *d_repls},
        strings_count,
        stream,
        mr);
  }();

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_re(strings_column_view const& strings,
                                   std::vector<std::string> const& patterns,
                                   strings_column_view const& repls,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_re(strings, patterns, repls, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
