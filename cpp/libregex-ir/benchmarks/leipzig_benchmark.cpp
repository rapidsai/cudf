/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "corpus_benchmark.hpp"

#include <nvbench/nvbench.cuh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

using regex_ir_benchmark::corpus_case;
using regex_ir_benchmark::corpus_source;

// expressions follow rust-leipzig/regex-performance src/main.c
constexpr std::array leipzig_cases{
  corpus_case{.name       = "01_twain",
              .family     = "literal",
              .expression = "Twain",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name             = "02_twain_ignore_case",
              .family           = "case_fold",
              .expression       = "Twain",
              .corpus           = corpus_source::Leipzig3200,
              .case_insensitive = true},
  corpus_case{.name       = "03_shing",
              .family     = "character_class",
              .expression = "[a-z]shing",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "04_huck_saw",
              .family     = "alternation",
              .expression = "Huck[a-zA-Z]+|Saw[a-zA-Z]+",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "05_word_nn",
              .family     = "assertion",
              .expression = R"REGEX(\b\w+nn\b)REGEX",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "06_negated_bounded",
              .family     = "bounded_repeat",
              .expression = "[a-q][^u-z]{13}x",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "07_names",
              .family     = "alternation",
              .expression = "Tom|Sawyer|Huckleberry|Finn",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name             = "08_names_ignore_case",
              .family           = "case_fold",
              .expression       = "Tom|Sawyer|Huckleberry|Finn",
              .corpus           = corpus_source::Leipzig3200,
              .case_insensitive = true},
  corpus_case{.name       = "09_optional_prefix",
              .family     = "bounded_repeat",
              .expression = ".{0,2}(Tom|Sawyer|Huckleberry|Finn)",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "10_required_prefix",
              .family     = "bounded_repeat",
              .expression = ".{2,4}(Tom|Sawyer|Huckleberry|Finn)",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "11_tom_river",
              .family     = "bounded_repeat",
              .expression = "Tom.{10,25}river|river.{10,25}Tom",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "12_word_ing",
              .family     = "repetition",
              .expression = "[a-zA-Z]+ing",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "13_bounded_ing",
              .family     = "bounded_repeat",
              .expression = R"REGEX(\s[a-zA-Z]{0,12}ing\s)REGEX",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "14_name_suffix",
              .family     = "captures",
              .expression = R"REGEX(([A-Za-z]awyer|[A-Za-z]inn)\s)REGEX",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "15_quoted_sentence",
              .family     = "bounded_repeat",
              .expression = R"REGEX(["'][^"']{0,30}[?!.]["'])REGEX",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name       = "16_unicode_symbols",
              .family     = "unicode_literal",
              .expression = "∞|✓",
              .corpus     = corpus_source::Leipzig3200},
  corpus_case{.name                  = "17_math_symbol_property",
              .family                = "unicode_property",
              .expression            = R"REGEX(\p{Sm})REGEX",
              .corpus                = corpus_source::Leipzig3200,
              .comparison_expression = R"REGEX([+<=>|~∞])REGEX"},
  corpus_case{.name       = "18_csv_field",
              .family     = "csv_repetition",
              .expression = R"REGEX((.*?,){13}z)REGEX",
              .corpus     = corpus_source::Leipzig3200}};

corpus_case const& get_case(nvbench::state& state)
{
  auto index = state.get_int64("Case");
  if (index < 1 || static_cast<std::size_t>(index) > leipzig_cases.size()) {
    throw std::invalid_argument("Case is outside the registered Leipzig range");
  }
  return leipzig_cases[static_cast<std::size_t>(index - 1)];
}

void regex_ir_leipzig(nvbench::state& state)
{
  regex_ir_benchmark::run_regex_ir(state, get_case(state));
}

void cudf_leipzig(nvbench::state& state) { regex_ir_benchmark::run_cudf(state, get_case(state)); }

}  // namespace

NVBENCH_BENCH(regex_ir_leipzig)
  .set_name("regex_ir/leipzig")
  .add_int64_axis("Case", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});

NVBENCH_BENCH(cudf_leipzig)
  .set_name("cudf/leipzig")
  .add_int64_axis("Case", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});
