/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/io/functions.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr size_t data_size         = 256 << 20;
constexpr cudf::size_type num_cols = 64;

namespace cudf_io = cudf::io;

class CsvRead : public cudf::benchmark {
};

void CSV_read(benchmark::State& state)
{
  auto const data_types = get_type_or_group(state.range(0));

  auto const tbl  = create_random_table(data_types, num_cols, table_size_bytes{data_size});
  auto const view = tbl->view();

  std::vector<char> out_buffer;
  out_buffer.reserve(data_size);
  cudf_io::write_csv_args args{cudf_io::sink_info(&out_buffer), view, "null", false, 1 << 30};
  cudf_io::write_csv(args);

  cudf_io::read_csv_args read_args(cudf_io::source_info(out_buffer.data(), out_buffer.size()));

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf_io::read_csv(read_args);
  }

  state.SetBytesProcessed(data_size * state.iterations());
}

// TODO: vary reader options instead of data profile here
#define CSV_RD_BENCHMARK_DEFINE(name, datatype)     \
  BENCHMARK_DEFINE_F(CsvRead, name)                 \
  (::benchmark::State & state) { CSV_read(state); } \
  BENCHMARK_REGISTER_F(CsvRead, name)               \
    ->Args({int32_t(datatype)})                     \
    ->Unit(benchmark::kMillisecond)                 \
    ->UseManualTime();

CSV_RD_BENCHMARK_DEFINE(integral, type_group_id::INTEGRAL);
CSV_RD_BENCHMARK_DEFINE(floats, type_group_id::FLOATING_POINT);
CSV_RD_BENCHMARK_DEFINE(timestamps, type_group_id::TIMESTAMP);
CSV_RD_BENCHMARK_DEFINE(string, cudf::type_id::STRING);
