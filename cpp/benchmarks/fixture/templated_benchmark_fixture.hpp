/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmark/benchmark.h>

namespace cudf {
/**
 * @brief Templated Google Benchmark with fixture
 *
 * Extends Google benchmarks to support templated Benchmarks with non-templated fixture class.
 *
 * The SetUp and TearDown methods is called before each templated benchmark function is run.
 * These methods are called automatically by Google Benchmark
 *
 * Example:
 *
 * @code
 * template <class T, class U>
 * void  my_benchmark(::benchmark::State& state) {
 *     std::vector<T> v1(state.range(0));
 *     std::vector<U> v2(state.range(0));
 *     for (auto _ : state) {
 *       // benchmark stuff
 *     }
 * }
 *
 * TEMPLATED_BENCHMARK_F(cudf::benchmark, my_benchmark, int, double)->Range(128, 512);
 * @endcode
 */
template <class Fixture>
class FunctionTemplateBenchmark : public Fixture {
 public:
  FunctionTemplateBenchmark(char const* name, ::benchmark::internal::Function* func)
    : Fixture(), func_(func)
  {
    this->SetName(name);
  }

  virtual void Run(::benchmark::State& st)
  {
    this->SetUp(st);
    this->BenchmarkCase(st);
    this->TearDown(st);
  }

 private:
  ::benchmark::internal::Function* func_;

 protected:
  virtual void BenchmarkCase(::benchmark::State& st) { func_(st); }
};

#define TEMPLATED_BENCHMARK_F(BaseClass, n, ...)                                           \
  BENCHMARK_PRIVATE_DECLARE(n) = (::benchmark::internal::RegisterBenchmarkInternal(        \
    new cudf::FunctionTemplateBenchmark<BaseClass>(#BaseClass "/" #n "<" #__VA_ARGS__ ">", \
                                                   n<__VA_ARGS__>)))

}  // namespace cudf
