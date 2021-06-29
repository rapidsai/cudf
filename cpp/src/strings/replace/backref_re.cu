/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "backref_re.cuh"

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <regex>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Parse the back-ref index and position values from a given replace format.
 *
 * The backref numbers are expected to be 1-based.
 *
 * Returns a modified string without back-ref indicators and a vector of backref
 * byte position pairs.
 * ```
 * Example:
 *    for input string:    'hello \2 and \1'
 *    the returned pairs:  (2,6),(1,11)
 *    returned string is:  'hello  and '
 * ```
 */
std::pair<std::string, std::vector<backref_type>> parse_backrefs(std::string const& repl)
{
  std::vector<backref_type> backrefs;
  std::string str = repl;  // make a modifiable copy
  std::smatch m;
  std::regex ex("(\\\\\\d+)");  // this searches for backslash-number(s); example "\1"
  std::string rtn;              // result without refs
  size_type byte_offset = 0;
  while (std::regex_search(str, m, ex)) {
    if (m.size() == 0) break;
    std::string const backref = m[0];
    size_type const position  = static_cast<size_type>(m.position(0));
    size_type const length    = static_cast<size_type>(backref.length());
    byte_offset += position;
    size_type const index = std::atoi(backref.c_str() + 1);  // back-ref index number
    CUDF_EXPECTS(index > 0, "Back-reference numbers must be greater than 0");
    rtn += str.substr(0, position);
    str = str.substr(position + length);
    backrefs.push_back({index, byte_offset});
  }
  if (!str.empty())  // add the remainder
    rtn += str;      // of the string
  return {rtn, backrefs};
}

}  // namespace

//
std::unique_ptr<column> replace_with_backrefs(
  strings_column_view const& strings,
  std::string const& pattern,
  std::string const& repl,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (strings.is_empty()) return make_empty_column(data_type{type_id::STRING});

  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(!repl.empty(), "Parameter repl must not be empty");

  auto d_strings = column_device_view::create(strings.parent(), stream);
  // compile regex into device object
  auto d_prog = reprog_device::create(pattern, get_character_flags_table(), strings.size(), stream);
  auto const regex_insts = d_prog->insts_counts();

  // parse the repl string for backref indicators
  auto const parse_result = parse_backrefs(repl);
  rmm::device_uvector<backref_type> backrefs(parse_result.second.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(backrefs.data(),
                           parse_result.second.data(),
                           sizeof(backref_type) * backrefs.size(),
                           cudaMemcpyHostToDevice,
                           stream.value()));
  string_scalar repl_scalar(parse_result.first, true, stream);
  string_view const d_repl_template = repl_scalar.value();

  using BackRefIterator = decltype(backrefs.begin());

  // create child columns
  auto [offsets, chars] = [&] {
    if (regex_insts <= RX_SMALL_INSTS) {
      return make_strings_children(
        backrefs_fn<BackRefIterator, RX_STACK_SMALL>{
          *d_strings, *d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
        strings.size(),
        stream,
        mr);
    } else if (regex_insts <= RX_MEDIUM_INSTS) {
      return make_strings_children(
        backrefs_fn<BackRefIterator, RX_STACK_MEDIUM>{
          *d_strings, *d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
        strings.size(),
        stream,
        mr);
    } else if (regex_insts <= RX_LARGE_INSTS) {
      return make_strings_children(
        backrefs_fn<BackRefIterator, RX_STACK_LARGE>{
          *d_strings, *d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
        strings.size(),
        stream,
        mr);
    } else {
      return make_strings_children(
        backrefs_fn<BackRefIterator, RX_STACK_ANY>{
          *d_strings, *d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
        strings.size(),
        stream,
        mr);
    }
  }();

  return make_strings_column(strings.size(),
                             std::move(offsets),
                             std::move(chars),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& strings,
                                              std::string const& pattern,
                                              std::string const& repl,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_with_backrefs(strings, pattern, repl, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
