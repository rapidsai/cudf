/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <benchmarks/common/benchmark_utilities.hpp>
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <memory>

class Reduction : public cudf::benchmark {};

template <typename type>
void BM_reduction(benchmark::State& state, std::unique_ptr<cudf::reduce_aggregation> const& agg)
{
  cudf::size_type const column_size{(cudf::size_type)state.range(0)};
  auto const dtype = cudf::type_to_id<type>();
  data_profile const profile =
    data_profile_builder().no_validity().distribution(dtype, distribution_id::UNIFORM, 0, 100);
  auto const input_column = create_random_column(dtype, row_count{column_size}, profile);

  cudf::data_type output_dtype =
    (agg->kind == cudf::aggregation::MEAN || agg->kind == cudf::aggregation::VARIANCE ||
     agg->kind == cudf::aggregation::STD)
      ? cudf::data_type{cudf::type_id::FLOAT64}
      : input_column->type();

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::reduce(*input_column, *agg, output_dtype);
  }

  // The benchmark takes a column and produces two scalars.
  set_items_processed(state, column_size + 1);
  set_bytes_processed(state, estimate_size(input_column->view()) + cudf::size_of(output_dtype));
}

#define concat(a, b, c) a##b##c
#define get_agg(op)     concat(cudf::make_, op, _aggregation<cudf::reduce_aggregation>())

// TYPE, OP
#define RBM_BENCHMARK_DEFINE(name, type, aggregation)             \
  BENCHMARK_DEFINE_F(Reduction, name)(::benchmark::State & state) \
  {                                                               \
    BM_reduction<type>(state, get_agg(aggregation));              \
  }                                                               \
  BENCHMARK_REGISTER_F(Reduction, name)                           \
    ->UseManualTime()                                             \
    ->Arg(10000)      /* 10k */                                   \
    ->Arg(100000)     /* 100k */                                  \
    ->Arg(1000000)    /* 1M */                                    \
    ->Arg(10000000)   /* 10M */                                   \
    ->Arg(100000000); /* 100M */

#define REDUCE_BENCHMARK_DEFINE(type, aggregation) \
  RBM_BENCHMARK_DEFINE(concat(type, _, aggregation), type, aggregation)

#define REDUCE_BENCHMARK_NUMERIC(aggregation)    \
  REDUCE_BENCHMARK_DEFINE(bool, aggregation);    \
  REDUCE_BENCHMARK_DEFINE(int8_t, aggregation);  \
  REDUCE_BENCHMARK_DEFINE(int32_t, aggregation); \
  REDUCE_BENCHMARK_DEFINE(int64_t, aggregation); \
  REDUCE_BENCHMARK_DEFINE(float, aggregation);   \
  REDUCE_BENCHMARK_DEFINE(double, aggregation);

REDUCE_BENCHMARK_NUMERIC(sum);
REDUCE_BENCHMARK_DEFINE(int32_t, product);
REDUCE_BENCHMARK_DEFINE(float, product);
REDUCE_BENCHMARK_DEFINE(int64_t, min);
REDUCE_BENCHMARK_DEFINE(double, min);
using cudf::timestamp_ms;
REDUCE_BENCHMARK_DEFINE(timestamp_ms, min);
REDUCE_BENCHMARK_DEFINE(int8_t, mean);
REDUCE_BENCHMARK_DEFINE(float, mean);
REDUCE_BENCHMARK_DEFINE(int32_t, variance);
REDUCE_BENCHMARK_DEFINE(double, variance);
REDUCE_BENCHMARK_DEFINE(int64_t, std);
REDUCE_BENCHMARK_DEFINE(float, std);
