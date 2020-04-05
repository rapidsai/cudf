#include <benchmark/benchmark.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cudf/types.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/transpose.hpp>
#include <memory>
#include <cudf/utilities/error.hpp>

template<bool use_validity, int shift_factor>
static void BM_transpose(benchmark::State& state)
{
    auto data = std::vector<int>(state.range(0), 0);
    auto validity = std::vector<bool>(state.range(0), 1);
    auto input_column = cudf::test::fixed_width_column_wrapper<int>(data.begin(),
                                                                    data.end(),
                                                                    validity.begin());
    
    cudf::column_view input_column_view = input_column;
    // auto input_column_views = std::vector<cudf::column_view>(state.range(1), input_column_view);
    auto input = cudf::table_view({ input_column_view, input_column_view });

    for (auto _ : state)
    {
        cuda_event_timer raii(state, true);
        auto output = cudf::transpose(input);
    }
}


class Transpose : public cudf::benchmark {};

#define TRANSPOSE_BM_BENCHMARK_DEFINE(name, use_validity, transpose_factor) \
    BENCHMARK_DEFINE_F(Transpose, name)(::benchmark::State & state) { \
        BM_transpose<use_validity, transpose_factor>(state); \
    } \
    BENCHMARK_REGISTER_F(Transpose, name) \
        ->RangeMultiplier(2) \
        ->Range(2, 2<<16) \
        ->UseManualTime() \
        ->Unit(benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(transpose_simple, false, 0);
