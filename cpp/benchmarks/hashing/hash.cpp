/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

class HashBenchmark : public cudf::benchmark {
};

enum contains_nulls { no_nulls, nulls };

static void BM_hash(benchmark::State& state, cudf::hash_id hid, contains_nulls has_nulls)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const data = create_random_table({cudf::type_id::INT64}, row_count{n_rows});
  if (has_nulls == contains_nulls::no_nulls)
    data->get_column(0).set_null_mask(rmm::device_buffer{}, 0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
    cudf::hash(data->view(), hid);
  }
}

#define concat(a, b, c) a##b##c

#define H_BENCHMARK_DEFINE(name, hid, n)                                            \
  BENCHMARK_DEFINE_F(HashBenchmark, name)                                           \
  (::benchmark::State & st) { BM_hash(st, cudf::hash_id::hid, contains_nulls::n); } \
  BENCHMARK_REGISTER_F(HashBenchmark, name)                                         \
    ->RangeMultiplier(4)                                                            \
    ->Ranges({{1 << 14, 1 << 24}})                                                  \
    ->UseManualTime()                                                               \
    ->Unit(benchmark::kMillisecond);

#define HASH_BENCHMARK_DEFINE(hid, n) H_BENCHMARK_DEFINE(concat(hid, _, n), hid, n)

HASH_BENCHMARK_DEFINE(HASH_MURMUR3, nulls)
HASH_BENCHMARK_DEFINE(HASH_SERIAL_MURMUR3, nulls)
HASH_BENCHMARK_DEFINE(HASH_SPARK_MURMUR3, nulls)
HASH_BENCHMARK_DEFINE(HASH_MD5, nulls)

HASH_BENCHMARK_DEFINE(HASH_MURMUR3, no_nulls)
HASH_BENCHMARK_DEFINE(HASH_SERIAL_MURMUR3, no_nulls)
HASH_BENCHMARK_DEFINE(HASH_SPARK_MURMUR3, no_nulls)
HASH_BENCHMARK_DEFINE(HASH_MD5, no_nulls)
