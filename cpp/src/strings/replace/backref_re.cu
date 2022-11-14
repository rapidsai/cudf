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

#include "backref_re.cuh"

#include <strings/regex/regex_program_impl.h>
#include <strings/regex/utilities.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <regex>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Return the capturing group index pattern to use with the given replacement string.
 *
 * Only two patterns are supported at this time `\d` and `${d}` where `d` is an integer in
 * the range 0-99. The `\d` pattern is returned by default unless no `\d` pattern is found in
 * the `repl` string,
 *
 * Reference: https://www.regular-expressions.info/refreplacebackref.html
 */
std::string get_backref_pattern(std::string_view repl)
{
  std::string const backslash_pattern = "\\\\(\\d+)";
  std::string const bracket_pattern   = "\\$\\{(\\d+)\\}";
  std::string const r{repl};
  std::smatch m;
  return std::regex_search(r, m, std::regex(backslash_pattern)) ? backslash_pattern
                                                                : bracket_pattern;
}
/**
 * @brief Parse the back-ref index and position values from a given replace format.
 *
 * The back-ref numbers are expected to be 1-based.
 *
 * Returns a modified string without back-ref indicators and a vector of back-ref
 * byte position pairs. These are used by the device code to build the output
 * string by placing the captured group elements into the replace format.
 *
 * For example, for input string 'hello \2 and \1' the returned `backref_type` vector
 * contains `[(2,6),(1,11)]` and the returned string is 'hello  and '.
 */
std::pair<std::string, std::vector<backref_type>> parse_backrefs(std::string_view repl,
                                                                 int const group_count)
{
  std::vector<backref_type> backrefs;
  std::string str{repl};  // make a modifiable copy
  std::smatch m;
  std::regex ex(get_backref_pattern(repl));
  std::string rtn;
  size_type byte_offset = 0;
  while (std::regex_search(str, m, ex) && !m.empty()) {
    // parse the back-ref index number
    size_type const index = static_cast<size_type>(std::atoi(std::string{m[1]}.c_str()));
    CUDF_EXPECTS(index >= 0 && index <= group_count,
                 "Group index numbers must be in the range 0 to group count");

    // store the new byte offset and index value
    size_type const position = static_cast<size_type>(m.position(0));
    byte_offset += position;
    backrefs.push_back({index, byte_offset});

    // update the output string
    rtn += str.substr(0, position);
    // remove the back-ref pattern to continue parsing
    str = str.substr(position + static_cast<size_type>(m.length(0)));
  }
  if (!str.empty())  // add the remainder
    rtn += str;      // of the string
  return {rtn, backrefs};
}

}  // namespace

//
std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                              regex_program const& prog,
                                              std::string_view replacement,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(!replacement.empty(), "Parameter replacement must not be empty");

  // create device object from regex_program
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  // parse the repl string for back-ref indicators
  auto group_count = std::min(99, d_prog->group_counts());  // group count should NOT exceed 99
  auto const parse_result = parse_backrefs(replacement, group_count);
  rmm::device_uvector<backref_type> backrefs =
    cudf::detail::make_device_uvector_async(parse_result.second, stream);
  string_scalar repl_scalar(parse_result.first, true, stream);
  string_view const d_repl_template = repl_scalar.value();

  auto const d_strings = column_device_view::create(input.parent(), stream);

  using BackRefIterator = decltype(backrefs.begin());
  auto children         = make_strings_children(
    backrefs_fn<BackRefIterator>{*d_strings, d_repl_template, backrefs.begin(), backrefs.end()},
    *d_prog,
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

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& strings,
                                              std::string_view pattern,
                                              std::string_view replacement,
                                              regex_flags const flags,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const h_prog = regex_program::create(pattern, flags, capture_groups::EXTRACT);
  return detail::replace_with_backrefs(
    strings, *h_prog, replacement, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& strings,
                                              regex_program const& prog,
                                              std::string_view replacement,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_with_backrefs(strings, prog, replacement, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
