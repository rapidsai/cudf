/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "regex_program_impl.h"

#include <cudf/strings/regex/regex_program.hpp>

#include <memory>
#include <string>

namespace cudf {
namespace strings {

std::unique_ptr<regex_program> regex_program::create(std::string_view pattern,
                                                     regex_flags flags,
                                                     capture_groups capture)
{
  auto p = new regex_program(pattern, flags, capture);
  return std::unique_ptr<regex_program>(p);
}

regex_program::~regex_program()                                = default;
regex_program::regex_program(regex_program&& other)            = default;
regex_program& regex_program::operator=(regex_program&& other) = default;

regex_program::regex_program(std::string_view pattern, regex_flags flags, capture_groups capture)
  : _pattern(pattern),
    _flags(flags),
    _impl(
      std::make_unique<regex_program_impl>(detail::reprog::create_from(pattern, flags, capture)))
{
}

std::string regex_program::pattern() const { return _pattern; }

regex_flags regex_program::flags() const { return _flags; }

capture_groups regex_program::capture() const { return _capture; }

int32_t regex_program::instructions_count() const { return _impl->prog.insts_count(); }

int32_t regex_program::groups_count() const { return _impl->prog.groups_count(); }

std::size_t regex_program::compute_working_memory_size(int32_t num_strings) const
{
  return detail::compute_working_memory_size(num_strings, instructions_count());
}

}  // namespace strings
}  // namespace cudf
