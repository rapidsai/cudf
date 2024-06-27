/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

std::unique_ptr<cudf::column> build_input_column(cudf::size_type n_rows,
                                                 cudf::size_type row_width,
                                                 int32_t hit_rate);

static void bench_find_string(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hit_rate  = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const api       = state.get_string("api");

  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  auto const stream = cudf::get_default_stream();
  auto const col    = build_input_column(n_rows, row_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  std::vector<std::string> h_targets({"5W", "5W43", "0987 5W43"});
  cudf::string_scalar target(h_targets[2]);
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const chars_size = input.chars_size(stream);
  state.add_element_count(chars_size, "chars_size");
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  if (api.substr(0, 4) == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  if (api == "find") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::find(input, target); });
  } else if (api == "find_multi") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::find_multiple(input, cudf::strings_column_view(targets));
    });
  } else if (api == "contains") {
    std::vector<std::string> match_targets{"123", "abc", "4567890", "DEFGHI", "5W43"};
    std::vector<cudf::string_scalar> targets;

    constexpr bool combine = true;
    constexpr int iters    = 20;
    for (int i = 0; i < iters; i++) {
      targets.emplace_back(cudf::string_scalar(match_targets[i % match_targets.size()]));
    }

    if constexpr (not combine) {
      state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        for (int i = 0; i < iters; i++) {
          [[maybe_unused]] auto output = cudf::strings::contains(input, targets[i]);
        }
      });
    } else {  // combine
      auto const output = cudf::strings::contains_multi_scalars(input, targets);
      for (std::size_t i = 0; i < output.size(); ++i) {
        auto const ref = cudf::strings::contains(input, targets[i]);
        cudf::test::detail::expect_columns_equal(ref->view(), output[i]->view());
      }

      state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        [[maybe_unused]] auto output = cudf::strings::contains_multi_scalars(input, targets);
      });
    }
  } else if (api == "starts_with") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::starts_with(input, target); });
  } else if (api == "ends_with") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::ends_with(input, target); });
  }
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_string")
  .add_string_axis("api", {"contains"})
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024})
  .add_int64_axis("num_rows", {260'000, 1'953'000, 16'777'216})
  .add_int64_axis("hit_rate", {20, 80});  // percentage
