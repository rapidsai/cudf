/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

class ReplaceNans : public cudf::benchmark {};

template <typename type>
static void BM_replace_nans(benchmark::State& state, bool include_nulls)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const dtype = cudf::type_to_id<type>();
  auto const input = create_random_column(dtype, row_count{n_rows});
  if (!include_nulls) input->set_null_mask(rmm::device_buffer{}, 0);

  auto zero = cudf::make_fixed_width_scalar<type>(0);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::replace_nans(*input, *zero);
  }
}

#define NANS_BENCHMARK_DEFINE(name, type, nulls)                        \
  BENCHMARK_DEFINE_F(ReplaceNans, name)                                 \
  (::benchmark::State & state) { BM_replace_nans<type>(state, nulls); } \
  BENCHMARK_REGISTER_F(ReplaceNans, name)                               \
    ->UseManualTime()                                                   \
    ->Arg(10000)      /* 10k */                                         \
    ->Arg(100000)     /* 100k */                                        \
    ->Arg(1000000)    /* 1M */                                          \
    ->Arg(10000000)   /* 10M */                                         \
    ->Arg(100000000); /* 100M */

NANS_BENCHMARK_DEFINE(float32_nulls, float, true);
NANS_BENCHMARK_DEFINE(float64_nulls, double, true);
NANS_BENCHMARK_DEFINE(float32_no_nulls, float, false);
NANS_BENCHMARK_DEFINE(float64_no_nulls, double, false);
