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
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <memory>
#include <random>

class Compaction : public cudf::benchmark {
};

template <typename Type>
void BM_compaction(benchmark::State& state, cudf::duplicate_keep_option keep)
{
  auto const n_rows = static_cast<cudf::size_type>(state.range(0));

  cudf::test::UniformRandomGenerator<long> rand_gen(0, 100);
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&rand_gen](auto row) { return rand_gen.generate(); });
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 100 == 0 ? false : true; });
  cudf::test::fixed_width_column_wrapper<Type, long> values(elements, elements + n_rows, valids);

  auto input_column = cudf::column_view(values);
  auto input_table  = cudf::table_view({input_column, input_column, input_column, input_column});

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::drop_duplicates(input_table, {0}, keep);
  }
}

#define concat(a, b, c) a##b##c
#define get_keep(op)    cudf::duplicate_keep_option::KEEP_##op

// TYPE, OP
#define RBM_BENCHMARK_DEFINE(name, type, keep)                     \
  BENCHMARK_DEFINE_F(Compaction, name)(::benchmark::State & state) \
  {                                                                \
    BM_compaction<type>(state, get_keep(keep));                    \
  }                                                                \
  BENCHMARK_REGISTER_F(Compaction, name)                           \
    ->UseManualTime()                                              \
    ->Arg(10000)    /* 10k */                                      \
    ->Arg(100000)   /* 100k */                                     \
    ->Arg(1000000)  /* 1M */                                       \
    ->Arg(10000000) /* 10M */

#define COMPACTION_BENCHMARK_DEFINE(type, keep) \
  RBM_BENCHMARK_DEFINE(concat(type, _, keep), type, keep)

COMPACTION_BENCHMARK_DEFINE(bool, NONE);
COMPACTION_BENCHMARK_DEFINE(int8_t, NONE);
COMPACTION_BENCHMARK_DEFINE(int32_t, NONE);
COMPACTION_BENCHMARK_DEFINE(int32_t, FIRST);
COMPACTION_BENCHMARK_DEFINE(int32_t, LAST);
using cudf::timestamp_ms;
COMPACTION_BENCHMARK_DEFINE(timestamp_ms, NONE);
COMPACTION_BENCHMARK_DEFINE(float, NONE);
