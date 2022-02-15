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
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

class ReductionScan : public cudf::benchmark {
};

template <typename type>
static void BM_reduction_scan(benchmark::State& state, bool include_nulls)
{
  // cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  size_t const size{(size_t)state.range(0)};
  auto const dtype = cudf::type_to_id<type>();
  data_profile table_data_profile;
  table_data_profile.set_distribution_params(dtype, distribution_id::UNIFORM, 0, 5);
  table_data_profile.set_null_frequency(0);
  // auto const table = create_random_table({dtype}, 1, row_count{n_rows}, table_data_profile);
  auto const table =
    create_random_table({dtype}, 1, table_size_bytes{size / 2}, table_data_profile);

  auto const new_tbl = cudf::repeat(table->view(), 2);
  if (!include_nulls) new_tbl->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
  cudf::column_view input(new_tbl->view().column(0));

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result =
      cudf::scan(input, cudf::make_dense_rank_aggregation(), cudf::scan_type::INCLUSIVE);
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

SCAN_BENCHMARK_DEFINE(int64_no_nulls, int32_t, false);
SCAN_BENCHMARK_DEFINE(list_no_nulls, cudf::list_view, false);
