/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/filling.hpp>

class Repeat : public cudf::benchmark {};

template <class TypeParam, bool nulls>
void BM_repeat(benchmark::State& state)
{
  auto const n_rows = static_cast<cudf::size_type>(state.range(0));
  auto const n_cols = static_cast<cudf::size_type>(state.range(1));

  auto const input_table =
    create_sequence_table(cycle_dtypes({cudf::type_to_id<TypeParam>()}, n_cols),
                          row_count{n_rows},
                          nulls ? std::optional<double>{1.0} : std::nullopt);
  // Create table view
  auto input = cudf::table_view(*input_table);

  // repeat counts
  using sizeT                = cudf::size_type;
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<sizeT>(), distribution_id::UNIFORM, 0, 3);
  auto repeat_count = create_random_column(cudf::type_to_id<sizeT>(), row_count{n_rows}, profile);

  // warm up
  auto output = cudf::repeat(input, *repeat_count);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::repeat(input, *repeat_count);
  }

  auto data_bytes =
    (input.num_columns() * input.num_rows() + output->num_columns() * output->num_rows()) *
    sizeof(TypeParam);
  auto null_bytes =
    nulls ? input.num_columns() * cudf::bitmask_allocation_size_bytes(input.num_rows()) +
              output->num_columns() * cudf::bitmask_allocation_size_bytes(output->num_rows())
          : 0;
  state.SetBytesProcessed(state.iterations() * (data_bytes + null_bytes));
}

#define REPEAT_BENCHMARK_DEFINE(name, type, nulls)                                                \
  BENCHMARK_DEFINE_F(Repeat, name)(::benchmark::State & state) { BM_repeat<type, nulls>(state); } \
  BENCHMARK_REGISTER_F(Repeat, name)                                                              \
    ->RangeMultiplier(8)                                                                          \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})                                                        \
    ->UseManualTime()                                                                             \
    ->Unit(benchmark::kMillisecond);

REPEAT_BENCHMARK_DEFINE(double_nulls, double, true);
REPEAT_BENCHMARK_DEFINE(double_no_nulls, double, false);
