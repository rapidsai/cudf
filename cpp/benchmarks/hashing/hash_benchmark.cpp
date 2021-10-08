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

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>

class HashBenchmark : public cudf::benchmark {
};

static void BM_hash(benchmark::State& state, cudf::hash_id hid)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const data = create_random_table({cudf::type_id::INT64}, 1, row_count{n_rows});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);
    cudf::hash(data->view(), hid);
  }
}

#define HASH_BENCHMARK_DEFINE(name)                               \
  BENCHMARK_DEFINE_F(HashBenchmark, name)                         \
  (::benchmark::State & st) { BM_hash(st, cudf::hash_id::name); } \
  BENCHMARK_REGISTER_F(HashBenchmark, name)                       \
    ->RangeMultiplier(4)                                          \
    ->Ranges({{1 << 14, 1 << 24}})                                \
    ->UseManualTime()                                             \
    ->Unit(benchmark::kMillisecond);

HASH_BENCHMARK_DEFINE(HASH_MURMUR3)
HASH_BENCHMARK_DEFINE(HASH_MD5)
HASH_BENCHMARK_DEFINE(HASH_SERIAL_MURMUR3)
HASH_BENCHMARK_DEFINE(HASH_SPARK_MURMUR3)
HASH_BENCHMARK_DEFINE(HASH_SHA1)
HASH_BENCHMARK_DEFINE(HASH_SHA224)
HASH_BENCHMARK_DEFINE(HASH_SHA256)
HASH_BENCHMARK_DEFINE(HASH_SHA384)
HASH_BENCHMARK_DEFINE(HASH_SHA512)
