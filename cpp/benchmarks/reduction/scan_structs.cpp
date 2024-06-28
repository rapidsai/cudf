/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/scan.hpp>

#include <nvbench/nvbench.cuh>

static constexpr cudf::size_type num_struct_members = 8;
static constexpr cudf::size_type max_int            = 100;
static constexpr cudf::size_type max_str_length     = 32;

static void nvbench_structs_scan(nvbench::state& state)
{
  auto const null_probability = [&] {
    auto const null_prob_val = state.get_float64("null_probability");
    return null_prob_val > 0 ? std::optional{null_prob_val} : std::nullopt;
  }();
  auto const size    = static_cast<cudf::size_type>(state.get_int64("data_size"));
  auto const profile = static_cast<data_profile>(
    data_profile_builder()
      .null_probability(null_probability)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, max_int)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length));

  auto data_table = create_random_table(
    cycle_dtypes({cudf::type_id::INT32, cudf::type_id::STRING}, num_struct_members),
    row_count{size},
    profile);
  auto [null_mask, null_count] = create_random_null_mask(size, null_probability);
  auto input                   = cudf::make_structs_column(
    size, std::move(data_table->release()), null_count, std::move(null_mask));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(std::move(input));
  cudf::table input_table{std::move(columns)};

  auto const agg         = cudf::make_min_aggregation<cudf::scan_aggregation>();
  auto const null_policy = static_cast<cudf::null_policy>(state.get_int64("null_policy"));
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  int64_t result_size = 0;
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    auto result = cudf::detail::scan_inclusive(
      input_table.view().column(0), *agg, null_policy, stream, rmm::mr::get_current_device_resource());
    timer.stop();

    // Estimating the result size will launch a kernel. Do not include it in measuring time.
    result_size += estimate_size(std::move(result));
  });

  state.add_element_count(input_table.num_rows());
  state.add_global_memory_reads(estimate_size(input_table.view()));
  state.add_global_memory_writes(result_size);

  set_throughputs(state);
}

NVBENCH_BENCH(nvbench_structs_scan)
  .set_name("structs_scan")
  .add_float64_axis("null_probability", {0, 0.1, 0.5, 0.9})
  .add_int64_axis("null_policy", {0, 1})
  .add_int64_axis("data_size",
                  {
                    10000,     // 10k
                    100000,    // 100k
                    1000000,   // 1M
                    10000000,  // 10M
                  });
