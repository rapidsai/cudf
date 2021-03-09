/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

class ReductionScan : public cudf::benchmark {
};

template <typename type>
static void BM_reduction_scan(benchmark::State& state, bool include_nulls)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const dtype = cudf::type_to_id<type>();
  auto const table = create_random_table({dtype}, 1, row_count{n_rows});
  if (!include_nulls) table->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
  cudf::column_view input(table->view().column(0));

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::scan(input, cudf::make_min_aggregation(), cudf::scan_type::INCLUSIVE);
  }
}

#define SCAN_BENCHMARK_DEFINE(name, type, nulls)                          \
  BENCHMARK_DEFINE_F(ReductionScan, name)                                 \
  (::benchmark::State & state) { BM_reduction_scan<type>(state, nulls); } \
  BENCHMARK_REGISTER_F(ReductionScan, name)                               \
    ->UseManualTime()                                                     \
    ->Arg(10000)      /* 10k */                                           \
    ->Arg(100000)     /* 100k */                                          \
    ->Arg(1000000)    /* 1M */                                            \
    ->Arg(10000000)   /* 10M */                                           \
    ->Arg(100000000); /* 100M */

SCAN_BENCHMARK_DEFINE(int8_no_nulls, int8_t, false);
SCAN_BENCHMARK_DEFINE(int32_no_nulls, int32_t, false);
SCAN_BENCHMARK_DEFINE(uint64_no_nulls, uint64_t, false);
SCAN_BENCHMARK_DEFINE(float_no_nulls, float, false);
SCAN_BENCHMARK_DEFINE(int16_nulls, int16_t, true);
SCAN_BENCHMARK_DEFINE(uint32_nulls, uint32_t, true);
SCAN_BENCHMARK_DEFINE(double_nulls, double, true);
