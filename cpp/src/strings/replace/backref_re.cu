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

#include "backref_re.cuh"

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
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
 * Returns a modified string without back-ref indicators.
 * ```
 * Example:
 *    for input string:    'hello \2 and \1'
 *    the returned pairs:  (2,6),(1,11)
 *    returned string is:  'hello  and '
 * ```
 */
std::string parse_backrefs(std::string const& repl, std::vector<backref_type>& backrefs)
{
  std::string str = repl;  // make a modifiable copy
  std::smatch m;
  std::regex ex("(\\\\\\d+)");  // this searches for backslash-number(s); example "\1"
  std::string rtn;              // result without refs
  size_type byte_offset = 0;
  while (std::regex_search(str, m, ex)) {
    if (m.size() == 0) break;
    backref_type item;
    std::string bref   = m[0];
    size_type position = static_cast<size_type>(m.position(0));
    size_type length   = static_cast<size_type>(bref.length());
    byte_offset += position;
    item.first = std::atoi(bref.c_str() + 1);  // back-ref index number
    CUDF_EXPECTS(item.first > 0, "Back-reference numbers must be greater than 0");
    item.second = byte_offset;  // position within the string
    rtn += str.substr(0, position);
    str = str.substr(position + length);
    backrefs.push_back(item);
  }
  if (!str.empty())  // add the remainder
    rtn += str;      // of the string
  return rtn;
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
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(stream, mr);

  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(!repl.empty(), "Parameter repl must not be empty");

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // compile regex into device object
  auto prog   = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_prog = *prog;
  auto regex_insts = d_prog.insts_counts();

  // parse the repl string for backref indicators
  std::vector<backref_type> h_backrefs;
  std::string repl_template = parse_backrefs(repl, h_backrefs);
  rmm::device_vector<backref_type> backrefs(h_backrefs);
  string_scalar repl_scalar(repl_template);
  string_view d_repl_template{repl_scalar.data(), repl_scalar.size()};

  // copy null mask
  auto null_mask  = cudf::detail::copy_bitmask(strings.parent(), stream, mr);
  auto null_count = strings.null_count();

  // create child columns
  children_pair children(nullptr, nullptr);
  // Each invocation is predicated on the stack size
  // which is dependent on the number of regex instructions
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS)) {
    children = make_strings_children(
      backrefs_fn<RX_STACK_SMALL>{
        d_strings, d_prog, d_repl_template, backrefs.begin(), backrefs.end()},
      strings_count,
      null_count,
      stream,
      mr);
  } else if (regex_insts <= RX_MEDIUM_INSTS)
    children = replace_with_backrefs_medium(
      d_strings, d_prog, d_repl_template, backrefs, null_count, stream, mr);
  else
    children = replace_with_backrefs_large(
      d_strings, d_prog, d_repl_template, backrefs, null_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask),
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
