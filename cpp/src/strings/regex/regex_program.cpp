/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "glushkov_regcomp.h"
#include "regex_program_impl.h"

#include <cudf/strings/regex/flags.hpp>
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

regex_program::~regex_program()                                         = default;
regex_program::regex_program(regex_program&& other) noexcept            = default;
regex_program& regex_program::operator=(regex_program&& other) noexcept = default;

regex_program::regex_program(std::string_view pattern, regex_flags flags, capture_groups capture)
  : _pattern(pattern),
    _flags(flags),
    _impl(
      std::make_unique<regex_program_impl>(detail::reprog::create_from(pattern, flags, capture)))
{
  // When the GLUSHKOV flag is requested, attempt to compile a Glushkov NFA.
  // build_glushkov_program returns nullptr if the pattern is ineligible
  // (e.g. has assertions or >64 positions), in which case we transparently
  // fall back to the Thompson NFA.  extract() bypasses Glushkov at the
  // device dispatch level (regex.inl) since Glushkov does not track groups.
  if (is_glushkov(flags)) { _impl->glushkov_prog = detail::build_glushkov_program(_impl->prog); }
}

std::string regex_program::pattern() const { return _pattern; }

regex_flags regex_program::flags() const { return _flags; }

capture_groups regex_program::capture() const { return _capture; }

int32_t regex_program::instructions_count() const { return _impl->prog.insts_count(); }

int32_t regex_program::groups_count() const { return _impl->prog.groups_count(); }

std::size_t regex_program::compute_working_memory_size(int32_t num_strings) const
{
  // Returns the Thompson NFA working memory size (the maximum possible).
  // On the device, reprog_device::working_memory_size() returns 0 for find-only
  // APIs when Glushkov is active; extract APIs disable Glushkov at creation time.
  return detail::compute_working_memory_size(num_strings, instructions_count());
}

}  // namespace strings
}  // namespace cudf
