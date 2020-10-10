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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <thrust/copy.h>
#include <thrust/detail/copy.h>
#include <thrust/iterator/constant_iterator.h>

#include <rmm/thrust_rmm_allocator.h>

__global__ void simple_kernel(uint8_t* result) {}

static void BM_avro(benchmark::State& state)
{
  auto const size = state.range(0);
  auto count_iter = thrust::make_constant_iterator<uint8_t>(0);

  rmm::device_vector<uint8_t> d_src(count_iter, count_iter + size);
  rmm::device_vector<uint8_t> d_dst(count_iter, count_iter + size);

  // launches
  for (auto _ : state) {
    {
      cuda_event_timer raii(state, true);
      thrust::copy(rmm::exec_policy(0)->on(0),  //
                   d_src.begin(),
                   d_src.end(),
                   d_dst.begin());
    }
    state.SetBytesProcessed(state.iterations() * size);
  }

  // constexpr int num_streams = 1000;

  // // setup
  // uint8_t* results;
  // cudaMalloc(&results, num_streams);

  // cudaStream_t streams[num_streams];
  // for (auto& stream : streams) { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }

  // int numBlocks = 1;
  // int n         = 32;
  // dim3 threadsPerBlock(n, n);

  // // launches
  // for (auto _ : state) {
  //   {
  //     cuda_event_timer raii(state, true);
  //     for (auto i = 0; i < num_streams; i++) {
  //       simple_kernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(&results[i]);
  //     }
  //   }
  // }

  // // tear down
  // for (auto& stream : streams) { cudaStreamDestroy(stream); }

  // cudaFree(results);
}

class AvroReaderBenchmark : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                                         \
  BENCHMARK_DEFINE_F(AvroReaderBenchmark, name)(::benchmark::State & state) { BM_avro(state); } \
  BENCHMARK_REGISTER_F(AvroReaderBenchmark, name)                                               \
    ->RangeMultiplier(32)                                                                       \
    ->Range(1 << 10, 1 << 30)                                                                   \
    ->UseManualTime()                                                                           \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(avro);
