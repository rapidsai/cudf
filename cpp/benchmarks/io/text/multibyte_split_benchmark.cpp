/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <thrust/transform.h>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/types.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <memory>

using cudf::test::fixed_width_column_wrapper;

static void BM_multibyte_split(benchmark::State& state)
{
  std::string host_input = "";
  int32_t num_chars      = state.range(0);

  for (auto i = 0; i < num_chars; i++) { host_input += "x"; }

  cudf::string_scalar input(host_input);

  auto delimiters = std::vector<std::string>({"😀", "😎", ",", "::"});

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::io::text::multibyte_split(input, delimiters);
  }

  state.SetBytesProcessed(state.iterations() * num_chars);
}

class MultibyteSplitBenchmark : public cudf::benchmark {
};

#define TRANSPOSE_BM_BENCHMARK_DEFINE(name)                                     \
  BENCHMARK_DEFINE_F(MultibyteSplitBenchmark, name)(::benchmark::State & state) \
  {                                                                             \
    BM_multibyte_split(state);                                                  \
  }                                                                             \
  BENCHMARK_REGISTER_F(MultibyteSplitBenchmark, name)                           \
    ->Range(1 << 15, 1 << 30)                                                   \
    ->UseManualTime()                                                           \
    ->Unit(benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(multibyte_split_simple);
