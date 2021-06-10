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

#include <cudf/row_conversion.hpp>
#include "cudf_test/column_utilities.hpp"

class RowConversion : public cudf::benchmark {
};

static void BM_to_row(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
/*  auto const table = create_random_table({cudf::type_id::INT8,
                                          cudf::type_id::INT32,
                                          cudf::type_id::INT16,
                                          cudf::type_id::INT64,
                                          cudf::type_id::INT32,
                                          cudf::type_id::BOOL8,
                                          cudf::type_id::UINT16,
                                          cudf::type_id::UINT8,
                                          cudf::type_id::UINT64},
                                         50,
                                         row_count{n_rows});*/
  auto const table = create_random_table({cudf::type_id::INT32},
  64,
  row_count{n_rows});

  cudf::size_type total_bytes = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto t = table->get_column(i).type();
    total_bytes += cudf::size_of(t);
  }

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);

//    auto rows = cudf::convert_to_rows(table->view());
    auto new_rows = cudf::convert_to_rows2(table->view());
  }

  state.SetBytesProcessed(state.iterations() * total_bytes * 2 * table->num_rows());
}

static void BM_from_row(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const table = create_random_table({cudf::type_id::INT8,
                                          cudf::type_id::INT32,
                                          cudf::type_id::INT16,
                                          cudf::type_id::INT64,
                                          cudf::type_id::INT32,
                                          cudf::type_id::BOOL8,
                                          cudf::type_id::UINT16,
                                          cudf::type_id::UINT8,
                                          cudf::type_id::UINT64},
                                         256,
                                         row_count{n_rows});
  /*  auto const table = create_random_table({cudf::type_id::INT32},
                                           4,
                                           row_count{n_rows});*/

  std::vector<cudf::data_type> schema;
  cudf::size_type total_bytes = 0;
  for (int i = 0; i < table->num_columns(); ++i) {
    auto t = table->get_column(i).type();
    schema.push_back(t);
    total_bytes += cudf::size_of(t);
  }

  auto rows = cudf::convert_to_rows(table->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);

    auto out = cudf::convert_from_rows(rows, schema);
  }

  state.SetBytesProcessed(state.iterations() * total_bytes * 2 * table->num_rows());
}

#define TO_ROW_CONVERSION_BENCHMARK_DEFINE(name) \
  BENCHMARK_DEFINE_F(RowConversion, name)        \
  (::benchmark::State & st) { BM_to_row(st); }   \
  BENCHMARK_REGISTER_F(RowConversion, name)      \
    ->RangeMultiplier(8)                         \
    ->Ranges({{1 << 6, 1 << 20}})               \
    ->UseManualTime()                            \
    ->Unit(benchmark::kMillisecond);

TO_ROW_CONVERSION_BENCHMARK_DEFINE(to_row_conversion)

#define FROM_ROW_CONVERSION_BENCHMARK_DEFINE(name) \
  BENCHMARK_DEFINE_F(RowConversion, name)          \
  (::benchmark::State & st) { BM_from_row(st); }   \
  BENCHMARK_REGISTER_F(RowConversion, name)        \
    ->RangeMultiplier(8)                           \
    ->Ranges({{1 << 6, 1 << 22}})                  \
    ->UseManualTime()                              \
    ->Unit(benchmark::kMillisecond);

FROM_ROW_CONVERSION_BENCHMARK_DEFINE(from_row_conversion)
