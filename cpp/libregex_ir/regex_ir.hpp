/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace regex_ir {

enum class character_mode : std::uint8_t {
  UTF8,
  BYTES,
};

struct compile_limits {
  std::size_t max_pattern_bytes = 1U << 20U;
  std::size_t max_nesting       = 256;
  std::size_t max_states        = 1U << 18U;
  std::size_t max_transitions   = 1U << 20U;
  std::size_t max_captures      = 256;
  std::uint32_t max_repeat      = 1000;
};

struct compile_options {
  bool case_insensitive : 1 = false;
  bool multiline        : 1 = false;
  bool dot_all          : 1 = false;
  bool ascii_classes    : 1 = true;
  bool extended_newline : 1 = false;
  character_mode characters = character_mode::UTF8;
  compile_limits limits     = compile_limits{};
};

enum class operation_kind : std::uint8_t {
  CONTAINS,
  MATCHES,
  COUNT,
  EXTRACT,
  FIND,
  REPLACE,
  SPLIT,
};

/**
 * @brief Compile a regular expression into operation-specialized NVVM IR
 *
 * `replacement` is required for `REPLACE` and rejected for every other
 * operation. Compilation failures are reported as `std::invalid_argument`.
 *
 * @param pattern Regular expression encoded as UTF-8 source bytes
 * @param operation Operation implemented by the generated entry point
 * @param replacement Replacement template for `REPLACE`
 * @param options Regex syntax, character-mode, and resource-limit options
 * @return Self-contained textual NVVM IR
 */
[[nodiscard]] std::string compile(
  std::string_view pattern,
  operation_kind operation,
  std::optional<std::string> replacement = std::nullopt,
  compile_options const& options          = {});

}  // namespace regex_ir
