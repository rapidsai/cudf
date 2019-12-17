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

#include <benchmark/benchmark.h>

#include <random>
#include <tests/utilities/column_wrapper.hpp>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

#include <cudf/detail/iterator.cuh> // include iterator header
// for reduction tests
#include <cub/device/device_reduce.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <thrust/device_vector.h>

template <typename T> T random_int(T min, T max) {
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T>
inline auto reduce_by_cub(OutputIterator result, InputIterator d_in,
                          int num_items, T init) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, result,
                            num_items, cudf::DeviceSum{}, init);

  // Allocate temporary storage
  RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, result,
                            num_items, cudf::DeviceSum{}, init);

  // Free temporary storage
  RMM_TRY(RMM_FREE(d_temp_storage, 0));

  return temp_storage_bytes;
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_cub(cudf::column_view &col,
                          rmm::device_vector<T> &result) {
  // std::cout << "raw stream cub: " << "\t";

  T init{0};
  auto begin = col.data<T>();
  int num_items = col.size();

  reduce_by_cub(result.begin(), begin, num_items, init);
  // T R;  cudaMemcpy(&R, result.data().get(), sizeof(T), cudaMemcpyDeviceToHost);
};

template <typename T, bool has_null>
void iterator_bench_cub(cudf::column_view &col, rmm::device_vector<T> &result) {
  // std::cout << "iterator cub " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col = cudf::column_device_view::create(col);
  int num_items = col.size();
  if (has_null) {
    auto begin = cudf::experimental::detail::make_null_replacement_iterator(
        *d_col, init);
    reduce_by_cub(result.begin(), begin, num_items, init);
  } else {
    auto begin = d_col->begin<T>();
    reduce_by_cub(result.begin(), begin, num_items, init);
  }
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_thrust(cudf::column_view &col,
                             rmm::device_vector<T> &result) {
  // std::cout << "raw stream thust: " << "\t\t";

  T init{0};
  auto d_in = col.data<T>();
  auto d_end = d_in + col.size();
  thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
}

template <typename T, bool has_null>
void iterator_bench_thrust(cudf::column_view &col,
                           rmm::device_vector<T> &result) {
  // std::cout << "iterator thust " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col = cudf::column_device_view::create(col);
  if (has_null) {
    auto d_in = cudf::experimental::detail::make_null_replacement_iterator(
        *d_col, init);
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  } else {
    auto d_in = d_col->begin<T>();
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  }
}

// -----------------------------------------------------------------------------
class Iterator : public cudf::benchmark {};

template <class TypeParam, int raw_or_iterator>
void BM_iterator(benchmark::State &state) {

  const cudf::size_type column_size{(cudf::size_type)state.range(0)};
  using T = TypeParam;
  auto num_gen = thrust::counting_iterator<cudf::size_type>(0);
  auto null_gen = thrust::make_transform_iterator(
      num_gen, [](cudf::size_type row) { return row % 2 == 0; });

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(
      num_gen, num_gen + column_size);
  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_T(
      num_gen, num_gen + column_size, null_gen);
  cudf::column_view hasnull_F = wrap_hasnull_F;
  cudf::column_view hasnull_T = wrap_hasnull_T;

  rmm::device_vector<T> dev_result(1, T{0});

  if (raw_or_iterator == 0) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      raw_stream_bench_cub<T>(hasnull_F, dev_result); // driven by raw pointer
    }
  } else if (raw_or_iterator == 1) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      iterator_bench_cub<T, false>(hasnull_F, dev_result); // driven by riterator without nulls
    }
  } else if (raw_or_iterator == 2) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      raw_stream_bench_thrust<T>(hasnull_F, dev_result); // driven by raw pointer
    }
  } else if (raw_or_iterator == 3) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      iterator_bench_thrust<T, false>(hasnull_F, dev_result); // driven by riterator without nulls
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())*column_size * sizeof(TypeParam));
}

#define ITER_BM_BENCHMARK_DEFINE(name, type, raw_or_iterator)                  \
  BENCHMARK_DEFINE_F(Iterator, name)(::benchmark::State & state) {             \
    BM_iterator<type, raw_or_iterator>(state);                                 \
  }                                                                            \
  BENCHMARK_REGISTER_F(Iterator, name)                                         \
      ->RangeMultiplier(10)                                                    \
      ->Range(1000, 10000000)                                                  \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMillisecond);

ITER_BM_BENCHMARK_DEFINE(double_cub_raw, double, 0);
ITER_BM_BENCHMARK_DEFINE(double_cub_iter, double, 1);
ITER_BM_BENCHMARK_DEFINE(double_thrust_raw, double, 2);
ITER_BM_BENCHMARK_DEFINE(double_thrust_iter, double, 3);