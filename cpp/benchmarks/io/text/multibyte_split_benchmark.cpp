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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_benchmark_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/file_utilities.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/types.hpp>

#include <thrust/transform.h>

#include <cstdio>
#include <fstream>
#include <memory>

using cudf::test::fixed_width_column_wrapper;

temp_directory const temp_dir("cudf_gbench");

static void BM_multibyte_split(benchmark::State& state)
{
  auto delimiters = std::vector<std::string>({"😀", "😎", ",", "::"});

  int32_t num_chars = state.range(0);
  auto host_input   = std::string(num_chars, 'x');
  auto device_input = cudf::string_scalar(host_input);

  auto temp_file_name = random_file_in_dir(temp_dir.path());

  close(mkstemp(const_cast<char*>(temp_file_name.data())));
  {
    auto temp_fostream = std::ofstream(temp_file_name, std::ofstream::out);
    temp_fostream << host_input;
    temp_fostream.close();
  }

  cudaDeviceSynchronize();

  auto source = cudf::io::text::make_source_from_file(temp_file_name);
  // auto source = cudf::io::text::make_source(device_input);
  // auto source = cudf::io::text::make_source(host_input);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::io::text::multibyte_split(*source, delimiters);
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
    ->Range(1 << 30, 1 << 30)                                                   \
    ->UseManualTime()                                                           \
    ->Unit(benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(multibyte_split_simple);
