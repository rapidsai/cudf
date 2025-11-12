/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <random>

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T>
inline auto reduce_by_cub(OutputIterator result, InputIterator d_in, int num_items, T init)
{
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(
    nullptr, temp_storage_bytes, d_in, result, num_items, cudf::DeviceSum{}, init);

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(temp_storage_bytes, cudf::get_default_stream());

  // Run reduction
  cub::DeviceReduce::Reduce(
    d_temp_storage.data(), temp_storage_bytes, d_in, result, num_items, cudf::DeviceSum{}, init);

  return temp_storage_bytes;
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_cub(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "raw stream cub: " << "\t";

  T init{0};
  auto begin    = col.data<T>();
  int num_items = col.size();

  reduce_by_cub(result.begin(), begin, num_items, init);
};

template <typename T, bool has_null>
void iterator_bench_cub(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "iterator cub " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col    = cudf::column_device_view::create(col);
  int num_items = col.size();
  if (has_null) {
    auto begin = cudf::detail::make_null_replacement_iterator(*d_col, init);
    reduce_by_cub(result.begin(), begin, num_items, init);
  } else {
    auto begin = d_col->begin<T>();
    reduce_by_cub(result.begin(), begin, num_items, init);
  }
}

// -----------------------------------------------------------------------------
template <typename T>
void raw_stream_bench_thrust(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "raw stream thust: " << "\t\t";

  T init{0};
  auto d_in  = col.data<T>();
  auto d_end = d_in + col.size();
  thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
}

template <typename T, bool has_null>
void iterator_bench_thrust(cudf::column_view& col, rmm::device_uvector<T>& result)
{
  // std::cout << "iterator thust " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_col = cudf::column_device_view::create(col);
  if (has_null) {
    auto d_in  = cudf::detail::make_null_replacement_iterator(*d_col, init);
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  } else {
    auto d_in  = d_col->begin<T>();
    auto d_end = d_in + col.size();
    thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{});
  }
}

// -----------------------------------------------------------------------------
class Iterator : public cudf::benchmark {};

template <class TypeParam, bool cub_or_thrust, bool raw_or_iterator>
void BM_iterator(benchmark::State& state)
{
  cudf::size_type const column_size{(cudf::size_type)state.range(0)};
  using T      = TypeParam;
  auto num_gen = thrust::counting_iterator<cudf::size_type>(0);

  cudf::test::fixed_width_column_wrapper<T> wrap_hasnull_F(num_gen, num_gen + column_size);
  cudf::column_view hasnull_F = wrap_hasnull_F;

  // Initialize dev_result to false
  auto dev_result = cudf::detail::make_zeroed_device_uvector<TypeParam>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    if (cub_or_thrust) {
      if (raw_or_iterator) {
        raw_stream_bench_cub<T>(hasnull_F, dev_result);  // driven by raw pointer
      } else {
        iterator_bench_cub<T, false>(hasnull_F, dev_result);  // driven by riterator without nulls
      }
    } else {
      if (raw_or_iterator) {
        raw_stream_bench_thrust<T>(hasnull_F, dev_result);  // driven by raw pointer
      } else {
        iterator_bench_thrust<T, false>(hasnull_F,
                                        dev_result);  // driven by riterator without nulls
      }
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * column_size *
                          sizeof(TypeParam));
}

#define ITER_BM_BENCHMARK_DEFINE(name, type, cub_or_thrust, raw_or_iterator) \
  BENCHMARK_DEFINE_F(Iterator, name)(::benchmark::State & state)             \
  {                                                                          \
    BM_iterator<type, cub_or_thrust, raw_or_iterator>(state);                \
  }                                                                          \
  BENCHMARK_REGISTER_F(Iterator, name)                                       \
    ->RangeMultiplier(10)                                                    \
    ->Range(1000, 10000000)                                                  \
    ->UseManualTime()                                                        \
    ->Unit(benchmark::kMillisecond);

ITER_BM_BENCHMARK_DEFINE(double_cub_raw, double, true, true);
ITER_BM_BENCHMARK_DEFINE(double_cub_iter, double, true, false);
ITER_BM_BENCHMARK_DEFINE(double_thrust_raw, double, false, true);
ITER_BM_BENCHMARK_DEFINE(double_thrust_iter, double, false, false);
