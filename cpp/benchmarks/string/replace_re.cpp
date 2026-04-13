/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <utility>
#include <vector>

// All patterns are Glushkov-compatible (no anchors ^ $ \b \B, all < 64 NFA positions).
// No capture groups in the patterns — for replace_with_backrefs the benchmark wraps
// each pattern in a capture group at runtime so \1 references the whole match.
// Match-rate estimates for 32-char random ASCII strings (chars 32-126, ~90% ASCII):
//   \d+              : ~97% of strings contain ≥1 digit run  (~3 matches/string)
//   [a-z]+[A-Z]+     : ~92% of strings contain ≥1 lower→upper transition (~2 matches)
//   [a-f]+|[0-5]+    : similar density to \d+ (~2 matches/string) -- exercises alternation
//   [a-z][0-9]{0,3}[A-Z]: ~92% via the zero-digit case alone -- exercises gap transitions
static std::vector<std::string> const patterns = {
  "\\d+",                  // 0: char class + quantifier (baseline)
  "[a-z]+[A-Z]+",          // 1: multi char-class sequence
  "[a-f]+|[0-5]+",         // 2: alternation (comparable density to \d+)
  "[a-z][0-9]{0,3}[A-Z]",  // 3: bounded repetition / gap transitions (7 positions)
  ".+[0-9]",               // 4: late-failure stress (~97% hit rate — quadratic for Glushkov):
              //    '.' matches all ASCII → inner loop runs full string from every start
  "[a-z]+Z",  // 5: late-failure + low hit rate (~23% on 32-char, ~79% on 256-char)
};

static void bench_replace(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width     = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width     = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const rtype         = state.get_string("type");
  auto const engine        = state.get_string("engine");

  // replace_with_backrefs requires capture groups; Glushkov doesn't support extract/backrefs
  if (engine == "glushkov" && rtype == "backref") {
    state.skip("backref replace — Glushkov doesn't support capture groups");
    return;
  }

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());

  auto flags = (engine == "glushkov") ? cudf::strings::regex_flags::GLUSHKOV
                                      : cudf::strings::regex_flags::DEFAULT;
  // Wrap in a capture group for backref replace so \1 references the whole match
  auto const pat =
    (rtype == "backref") ? "(" + patterns[pattern_index] + ")" : patterns[pattern_index];
  auto program = cudf::strings::regex_program::create(pat, flags);

  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  if (rtype == "backref") {
    auto replacement = std::string("#\\1X");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_with_backrefs(input, *program, replacement);
    });
  } else {
    auto replacement = std::string_view("77");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_re(input, *program, replacement);
    });
  }
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace_re")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("pattern", {0, 1, 2, 3, 4, 5})
  .add_string_axis("type", {"replace", "backref"})
  .add_string_axis("engine", {"thompson", "glushkov"});
