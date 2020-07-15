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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <memory>
#include <random>

class Reduction : public cudf::benchmark {
};

template <typename type>
void BM_reduction(benchmark::State& state, std::unique_ptr<cudf::aggregation> const& agg)
{
  using wrapper = cudf::test::fixed_width_column_wrapper<type>;
  const cudf::size_type column_size{(cudf::size_type)state.range(0)};

  cudf::test::UniformRandomGenerator<long> rand_gen(0, 100);
  auto data_it = cudf::test::make_counting_transform_iterator(
    0, [&rand_gen](cudf::size_type row) { return rand_gen.generate(); });
  wrapper values(data_it, data_it + column_size);

  auto input_column = cudf::column_view(values);
  cudf::data_type output_dtype =
    (agg->kind == cudf::aggregation::MEAN || agg->kind == cudf::aggregation::VARIANCE ||
     agg->kind == cudf::aggregation::STD)
      ? cudf::data_type{cudf::type_id::FLOAT64}
      : input_column.type();

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::reduce(input_column, agg, output_dtype);
  }
}

#define concat(a, b, c) a##b##c
#define get_agg(op) concat(cudf::make_, op, _aggregation())

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
REDUCE_BENCHMARK_NUMERIC(product);
REDUCE_BENCHMARK_NUMERIC(min);
using cudf::timestamp_ms;
REDUCE_BENCHMARK_DEFINE(timestamp_ms, min);
REDUCE_BENCHMARK_NUMERIC(mean);
REDUCE_BENCHMARK_NUMERIC(variance);
REDUCE_BENCHMARK_NUMERIC(std);
