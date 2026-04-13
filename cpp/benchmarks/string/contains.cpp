/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <utility>
#include <vector>

// create_string_column generates data from 10 hardcoded strings where only row 0
// ("123 abc 4567890 DEFGHI 0987 5W43") is scattered at exactly hit_rate%.
// All patterns below match ONLY row 0 (via the unique "5W" substring), so
// the hit_rate parameter correctly controls the match frequency for every pattern.
// longer pattern lengths demand more working memory per string
// patterns 0 and 1 contain anchors (^ $) so Glushkov falls back to Thompson for those
static std::vector<std::string> const patterns = {
  "^\\d+ [a-z]+",                  // 0: anchor pattern (anchors ^ $)
  "[A-Z ]+\\d+ +\\d+[A-Z]+\\d+$",  // 1: anchor pattern (anchors ^ $)
  "5W43",                          // 2: simple literal (baseline)
  "5[A-Z]\\d+",                    // 3: char class + quantifier (3 positions)
  "5W43|X9Z8",                     // 4: alternation (8 positions; only "5W43" branch matches)
  "5W4{1,3}",                      // 5: bounded repetition (5 positions)
  "(?:5W){1,2}",                   // 6: non-capturing group + bounded rep (4 positions)
  "5.4.",                          // 7: dot wildcard (4 positions)
  ".+5W",                          // 8: late-failure stress (dot prefix):
           //    '.' matches everything → phase 1 state never dies until "5W" found;
           //    only row 0 has "5W" → hit_rate controls match frequency correctly
};

static void bench_contains(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width     = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const hit_rate      = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const engine        = state.get_string("engine");

  // Patterns 0-1 contain anchors (^ $) which Glushkov doesn't support
  if (engine == "glushkov" && pattern_index <= 1) {
    state.skip("anchor pattern — Glushkov falls back to Thompson");
    return;
  }

  auto col   = create_string_column(num_rows, row_width, hit_rate);
  auto input = cudf::strings_column_view(col->view());

  auto flags   = (engine == "glushkov") ? cudf::strings::regex_flags::GLUSHKOV
                                        : cudf::strings::regex_flags::DEFAULT;
  auto program = cudf::strings::regex_program::create(patterns[pattern_index], flags);

  state.add_global_memory_reads<nvbench::int8_t>(col->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(input.size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::contains_re(input, *program); });
}

NVBENCH_BENCH(bench_contains)
  .set_name("contains")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {50, 100})  // percentage
  .add_int64_axis("pattern", {0, 1, 2, 3, 4, 5, 6, 7, 8})
  .add_string_axis("engine", {"thompson", "glushkov"});
