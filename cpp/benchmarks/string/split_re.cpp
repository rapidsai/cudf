/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/split/split_re.hpp>
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
  "[a-z]+[A-Z]+",          // 1: multi char-class sequence
  "[a-f]+|[0-5]+",         // 2: alternation (comparable density to \d+)
  "[a-z][0-9]{0,3}[A-Z]",  // 3: bounded repetition / gap transitions (7 positions)
  ".+[0-9]",               // 4: late-failure stress (~97% hit rate — quadratic for Glushkov):
              //    '.' matches all ASCII → inner loop runs full string from every start
  "[a-z]+Z",  // 5: late-failure + low hit rate (~23% on 32-char, ~79% on 256-char)
};

static void bench_split(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width     = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width     = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const engine        = state.get_string("engine");

  auto flags = (engine == "glushkov") ? cudf::strings::regex_flags::GLUSHKOV
                                      : cudf::strings::regex_flags::DEFAULT;
  auto prog  = cudf::strings::regex_program::create(patterns[pattern_index], flags);

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(data_size);  // all bytes are written

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::strings::split_record_re(input, *prog);
  });
}

NVBENCH_BENCH(bench_split)
  .set_name("split_re")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("pattern", {0, 1, 2, 3, 4, 5})
  .add_string_axis("engine", {"thompson", "glushkov"});
