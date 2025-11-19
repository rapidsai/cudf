/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

class ReplaceClamp : public cudf::benchmark {};

template <typename type>
static void BM_clamp(benchmark::State& state, bool include_nulls)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const dtype = cudf::type_to_id<type>();
  auto const input = create_random_column(dtype, row_count{n_rows});
  if (!include_nulls) input->set_null_mask(rmm::device_buffer{}, 0);

  auto [low_scalar, high_scalar] = cudf::minmax(*input);

  // set the clamps 2 in from the min and max
  {
    using ScalarType = cudf::scalar_type_t<type>;
    auto lvalue      = static_cast<ScalarType*>(low_scalar.get());
    auto hvalue      = static_cast<ScalarType*>(high_scalar.get());

    // super heavy clamp
    auto mid = lvalue->value() + (hvalue->value() - lvalue->value()) / 2;
    lvalue->set_value(mid - 10);
    hvalue->set_value(mid + 10);
  }

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::clamp(*input, *low_scalar, *high_scalar);
  }
}

#define CLAMP_BENCHMARK_DEFINE(name, type, nulls)                \
  BENCHMARK_DEFINE_F(ReplaceClamp, name)                         \
  (::benchmark::State & state) { BM_clamp<type>(state, nulls); } \
  BENCHMARK_REGISTER_F(ReplaceClamp, name)                       \
    ->UseManualTime()                                            \
    ->Arg(10000)      /* 10k */                                  \
    ->Arg(100000)     /* 100k */                                 \
    ->Arg(1000000)    /* 1M */                                   \
    ->Arg(10000000)   /* 10M */                                  \
    ->Arg(100000000); /* 100M */

CLAMP_BENCHMARK_DEFINE(int8_no_nulls, int8_t, false);
CLAMP_BENCHMARK_DEFINE(int32_no_nulls, int32_t, false);
CLAMP_BENCHMARK_DEFINE(uint64_no_nulls, uint64_t, false);
CLAMP_BENCHMARK_DEFINE(float_no_nulls, float, false);
CLAMP_BENCHMARK_DEFINE(int16_nulls, int16_t, true);
CLAMP_BENCHMARK_DEFINE(uint32_nulls, uint32_t, true);
CLAMP_BENCHMARK_DEFINE(double_nulls, double, true);
