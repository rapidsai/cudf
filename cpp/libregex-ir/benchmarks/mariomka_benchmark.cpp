/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "corpus_benchmark.hpp"

#include <nvbench/nvbench.cuh>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace {

using regex_ir_benchmark::corpus_case;
using regex_ir_benchmark::corpus_source;

// patterns follow mariomka/regex-benchmark at the pinned corpus revision
constexpr std::array mariomka_cases{
  corpus_case{.name       = "01_email",
              .family     = "email",
              .expression = R"REGEX([\w\.+-]+@[\w\.-]+\.[\w\.-]+)REGEX",
              .corpus     = corpus_source::MariomkaInput},
  corpus_case{.name       = "02_uri",
              .family     = "uri",
              .expression = R"REGEX([\w]+:\/\/[^\/\s?#]+[^\s?#]+(?:\?[^\s#]*)?(?:#[^\s]*)?)REGEX",
              .corpus     = corpus_source::MariomkaInput},
  corpus_case{
    .name   = "03_ipv4",
    .family = "ipv4",
    .expression =
      R"REGEX((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]))REGEX",
    .corpus = corpus_source::MariomkaInput}};

corpus_case const& get_case(nvbench::state& state)
{
  auto index = state.get_int64("Case");
  if (index < 1 || static_cast<std::size_t>(index) > mariomka_cases.size()) {
    throw std::invalid_argument("Case is outside the registered mariomka range");
  }
  return mariomka_cases[static_cast<std::size_t>(index - 1)];
}

void regex_ir_mariomka(nvbench::state& state)
{
  regex_ir_benchmark::run_regex_ir(state, get_case(state));
}

void cudf_mariomka(nvbench::state& state) { regex_ir_benchmark::run_cudf(state, get_case(state)); }

}  // namespace

NVBENCH_BENCH(regex_ir_mariomka)
  .set_name("regex_ir/mariomka")
  .add_int64_axis("Case", {1, 2, 3})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});

NVBENCH_BENCH(cudf_mariomka)
  .set_name("cudf/mariomka")
  .add_int64_axis("Case", {1, 2, 3})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});
