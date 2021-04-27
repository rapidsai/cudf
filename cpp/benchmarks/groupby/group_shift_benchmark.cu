/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <benchmark/benchmark.h>

#include <random>

class Groupby : public cudf::benchmark {
};

// TODO: put it in a struct so `uniform` can be remade with different min, max
template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

void BM_group_shift(benchmark::State& state)
{
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  const cudf::size_type column_size{(cudf::size_type)state.range(0)};
  const int num_groups = 100;

  auto data_it = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type row) { return random_int(0, num_groups); });

  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size);

  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));

  cudf::size_type offset =
    static_cast<cudf::size_type>(column_size / float(num_groups) * 0.5);  // forward shift half way
  // null fill value
  auto fill_value = cudf::make_default_constructed_scalar(cudf::data_type(cudf::type_id::INT64));
  // non null fill value
  // auto fill_value = cudf::make_fixed_width_scalar(static_cast<int64_t>(42));

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = gb_obj.shift(vals, offset, *fill_value);
  }
}

BENCHMARK_DEFINE_F(Groupby, Shift)(::benchmark::State& state) { BM_group_shift(state); }

BENCHMARK_REGISTER_F(Groupby, Shift)
  ->Arg(1000000)
  ->Arg(10000000)
  ->Arg(100000000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
