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

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <benchmark/benchmark.h>
#include <chrono>
#include <random>

namespace {

constexpr cudf::size_type hundredM      = 1e8;
constexpr cudf::size_type tenM          = 1e7;
constexpr cudf::size_type tenK          = 1e4;
constexpr cudf::size_type fifty_percent = 50;

void percent_range(benchmark::internal::Benchmark* b)
{
  b->Unit(benchmark::kMillisecond);
  for (int percent = 0; percent <= 100; percent += 10) b->Args({hundredM, percent});
}

void size_range(benchmark::internal::Benchmark* b)
{
  b->Unit(benchmark::kMillisecond);
  for (int size = tenK; size <= hundredM; size *= 10) b->Args({size, fifty_percent});
}

template <typename T>
T random_int(T min, T max)
{
  static unsigned const seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

template <typename T>
void calculate_bandwidth(benchmark::State& state, cudf::size_type num_columns)
{
  cudf::size_type const column_size{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const percent_true{static_cast<cudf::size_type>(state.range(1))};

  float const fraction                  = percent_true / 100.f;
  cudf::size_type const column_size_out = fraction * column_size;
  int64_t const mask_size =
    sizeof(bool) * column_size + cudf::bitmask_allocation_size_bytes(column_size);
  int64_t const validity_bytes_in = (fraction >= 1.0f / 32)
                                      ? cudf::bitmask_allocation_size_bytes(column_size)
                                      : 4 * column_size_out;
  int64_t const validity_bytes_out = cudf::bitmask_allocation_size_bytes(column_size_out);
  int64_t const column_bytes_out   = sizeof(T) * column_size_out;
  int64_t const column_bytes_in    = column_bytes_out;  // we only read unmasked inputs

  int64_t const bytes_read =
    (column_bytes_in + validity_bytes_in) * num_columns +  // reading columns
    mask_size;                                             // reading boolean mask
  int64_t const bytes_written =
    (column_bytes_out + validity_bytes_out) * num_columns;  // writing columns

  state.SetItemsProcessed(state.iterations() * column_size * num_columns);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_read + bytes_written);
}

}  // namespace

template <class T>
void BM_apply_boolean_mask(benchmark::State& state, cudf::size_type num_columns)
{
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;
  using mask_wrapper = cudf::test::fixed_width_column_wrapper<bool>;

  const cudf::size_type column_size{static_cast<cudf::size_type>(state.range(0))};
  const cudf::size_type percent_true{static_cast<cudf::size_type>(state.range(1))};

  std::vector<wrapper> columns;

  std::vector<T> data(column_size);
  std::vector<bool> validity(column_size, true);

  std::iota(data.begin(), data.end(), 0);

  for (int i = 0; i < num_columns; i++) {
    columns.emplace_back(data.cbegin(), data.cend(), validity.cbegin());
  }

  std::vector<bool> mask_data(column_size);
  std::generate_n(
    mask_data.begin(), column_size, [&]() { return random_int(0, 100) < percent_true; });
  mask_wrapper mask(mask_data.begin(), mask_data.end());

  std::vector<cudf::column_view> column_views(num_columns);

  std::transform(columns.begin(), columns.end(), column_views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });
  cudf::table_view source_table{column_views};

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto result = cudf::apply_boolean_mask(source_table, mask);
  }

  calculate_bandwidth<T>(state, num_columns);
}

template <class T>
class ApplyBooleanMask : public cudf::benchmark {
 public:
  using TypeParam = T;
};

#define ABM_BENCHMARK_DEFINE(name, type, n_columns)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(ApplyBooleanMask, name, type)(::benchmark::State & st) \
  {                                                                                  \
    BM_apply_boolean_mask<TypeParam>(st, n_columns);                                 \
  }

ABM_BENCHMARK_DEFINE(float_1_col, float, 1);
ABM_BENCHMARK_DEFINE(float_2_col, float, 2);
ABM_BENCHMARK_DEFINE(float_4_col, float, 4);

// shmoo 1, 2, 4 column float across percentage true
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_1_col)->Apply(percent_range);
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_2_col)->Apply(percent_range);
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_4_col)->Apply(percent_range);

// shmoo 1, 2, 4 column float across column sizes with 50% true
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_1_col)->Apply(size_range);
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_2_col)->Apply(size_range);
BENCHMARK_REGISTER_F(ApplyBooleanMask, float_4_col)->Apply(size_range);

// spot benchmark other types
ABM_BENCHMARK_DEFINE(int8_1_col, int8_t, 1);
ABM_BENCHMARK_DEFINE(int16_1_col, int16_t, 1);
ABM_BENCHMARK_DEFINE(int32_1_col, int32_t, 1);
ABM_BENCHMARK_DEFINE(int64_1_col, int64_t, 1);
ABM_BENCHMARK_DEFINE(double_1_col, double, 1);
BENCHMARK_REGISTER_F(ApplyBooleanMask, int8_1_col)->Args({tenM, fifty_percent});
BENCHMARK_REGISTER_F(ApplyBooleanMask, int16_1_col)->Args({tenM, fifty_percent});
BENCHMARK_REGISTER_F(ApplyBooleanMask, int32_1_col)->Args({tenM, fifty_percent});
BENCHMARK_REGISTER_F(ApplyBooleanMask, int64_1_col)->Args({tenM, fifty_percent});
BENCHMARK_REGISTER_F(ApplyBooleanMask, double_1_col)->Args({tenM, fifty_percent});
