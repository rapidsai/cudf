/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <string_view>

namespace regex_ir_benchmark {

/**
 * @brief Complete source dataset used by an imported regex benchmark case
 */
enum class corpus_source : std::uint8_t {
  OpenRestyAlphabet       = 0,  ///< Full 25 MiB `abc.txt` generation recipe
  OpenRestyRandomAlphabet = 1,  ///< Full 10 MiB `rand-abc.txt` generation recipe
  OpenRestyDelimiter      = 2,  ///< Full 10 MiB `delim.txt` generation recipe
  Mtent12                 = 3,  ///< OpenResty/Boost 20,045,118-byte Mark Twain file
  Mtent12Prefix50K        = 4,  ///< First 50,000 bytes used by Boost's medium search
  Leipzig3200             = 5,  ///< Rust Leipzig's complete 16,013,977-byte input
  BoostCrc                = 6,  ///< Complete Boost 1.41 `boost/crc.hpp`
  BoostLibraries          = 7,  ///< Complete Boost 1.41 `libs/libraries.htm`
  MariomkaInput           = 8,  ///< Complete Learn X in Y Minutes concatenation
  Inline                  = 9,  ///< Exact scalar text stored in `inline_text`
};

/**
 * @brief One regex and corpus recipe in an external benchmark suite
 */
struct corpus_case {
  std::string_view name                  = "";  ///< Stable benchmark case name
  std::string_view family                = "";  ///< Pattern-feature family
  std::string_view expression            = "";  ///< Regex expression
  corpus_source corpus                   = corpus_source::OpenRestyAlphabet;  ///< Source dataset
  std::string_view inline_text           = "";     ///< Complete input for an inline scalar case
  std::string_view comparison_expression = "";     ///< Equivalent expression for cuDF syntax
  bool case_insensitive : 1              = false;  ///< Enable case-insensitive matching
  bool multiline        : 1              = false;  ///< Enable line-oriented anchors
};

/**
 * @brief Run one corpus case through the Regex IR NVVM implementation
 *
 * @param state NVBench state containing Rows, Columns, and MaxStringBytes axes
 * @param benchmark Regex and corpus recipe
 */
void run_regex_ir(nvbench::state& state, corpus_case const& benchmark);

/**
 * @brief Run one corpus case through the cuDF regex implementation
 *
 * @param state NVBench state containing Rows, Columns, and MaxStringBytes axes
 * @param benchmark Regex and corpus recipe
 */
void run_cudf(nvbench::state& state, corpus_case const& benchmark);

}  // namespace regex_ir_benchmark
