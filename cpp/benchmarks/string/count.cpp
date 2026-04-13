/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <utility>
#include <vector>

// All patterns are Glushkov-compatible (no anchors ^ $ \b \B, all < 64 NFA positions).
// Match-rate estimates for 32-char random ASCII strings (chars 32-126, ~90% ASCII):
//   \d+              : ~97% of strings contain ≥1 digit run  (~3 matches/string)
//   [a-z]+[A-Z]+     : ~92% of strings contain ≥1 lower→upper transition (~2 matches)
//   [a-f]+|[0-5]+    : similar density to \d+ (~2 matches/string) -- exercises alternation
//   [a-z][0-9]{0,3}[A-Z]: ~92% via the zero-digit case alone -- exercises gap transitions
static std::vector<std::string> const patterns = {
  "\\d+",                  // 0: char class + quantifier (baseline)
  "a",                     // 1: simple literal
  "[a-z]+[A-Z]+",          // 2: multi char-class sequence
  "[a-f]+|[0-5]+",         // 3: alternation (comparable density to \d+)
  "[a-z][0-9]{0,3}[A-Z]",  // 4: bounded repetition / gap transitions (7 positions)
  ".+[0-9]",               // 5: late-failure stress (~97% hit rate, ~1 match/string):
                           //    '.' matches all ASCII → phase 1 state never dies until digit
                           //    found; O(n) two-phase search for both engines
  "[a-z]+Z",               // 6: late-failure + low hit rate (~23% on 32-char strings,
                           //    ~79% on 256-char strings); Glushkov skips ~73% of start
                           //    positions via reach filter
};

static void bench_count(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width     = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width     = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const engine        = state.get_string("engine");

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));

  auto const pattern = patterns[pattern_index];
  auto const flags   = (engine == "glushkov") ? cudf::strings::regex_flags::GLUSHKOV
                                              : cudf::strings::regex_flags::DEFAULT;
  auto prog          = cudf::strings::regex_program::create(pattern, flags);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto data_size = table->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int32_t>(input.size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::strings::count_re(input, *prog); });
}

NVBENCH_BENCH(bench_count)
  .set_name("count")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("pattern", {0, 1, 2, 3, 4, 5, 6})
  .add_string_axis("engine", {"thompson", "glushkov"});
