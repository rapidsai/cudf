#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <memory>
#include <tests/utilities/column_wrapper.hpp>

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  cudaStream_t stream = 0, rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  auto s = new ScalarType(0, false, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  T value,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  auto s = new ScalarType(value, true, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T>
struct value_func {
  T* data;
  cudf::size_type offset;

  __device__ T operator()(int idx) { return data[idx - offset]; }
};

struct validity_func {
  cudf::size_type size;
  cudf::size_type offset;

  __device__ bool operator()(int idx)
  {
    auto source_idx = idx - offset;
    return source_idx < 0 || source_idx >= size;
  }
};

template <bool use_validity, int shift_factor>
static void BM_shift(benchmark::State& state)
{
  cudf::size_type size   = state.range(0);
  cudf::size_type offset = size * (static_cast<double>(shift_factor) / 100.0);
  auto idx_begin         = thrust::make_counting_iterator<cudf::size_type>(0);
  auto idx_end           = thrust::make_counting_iterator<cudf::size_type>(size);

  auto input = use_validity
                 ? cudf::test::fixed_width_column_wrapper<int>(
                     idx_begin,
                     idx_end,
                     thrust::make_transform_iterator(idx_begin, [](auto idx) { return true; }))
                 : cudf::test::fixed_width_column_wrapper<int>(idx_begin, idx_end);

  auto fill = use_validity ? make_scalar<int>() : make_scalar<int>(777);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::shift(input, offset, *fill);
  }
}

class Shift : public cudf::benchmark {
};

#define SHIFT_BM_BENCHMARK_DEFINE(name, use_validity, shift_factor) \
  BENCHMARK_DEFINE_F(Shift, name)(::benchmark::State & state)       \
  {                                                                 \
    BM_shift<use_validity, shift_factor>(state);                    \
  }                                                                 \
  BENCHMARK_REGISTER_F(Shift, name)                                 \
    ->RangeMultiplier(32)                                           \
    ->Range(1 << 10, 1 << 30)                                       \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

SHIFT_BM_BENCHMARK_DEFINE(shift_zero, false, 0);
SHIFT_BM_BENCHMARK_DEFINE(shift_zero_nullable_out, true, 0);

SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent, false, 10);
SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent_nullable_out, true, 10);

SHIFT_BM_BENCHMARK_DEFINE(shift_half, false, 50);
SHIFT_BM_BENCHMARK_DEFINE(shift_half_nullable_out, true, 50);

SHIFT_BM_BENCHMARK_DEFINE(shift_full, false, 100);
SHIFT_BM_BENCHMARK_DEFINE(shift_full_nullable_out, true, 100);
