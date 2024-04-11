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

#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

class CopyIfElse : public cudf::benchmark {};

template <class TypeParam>
static void BM_copy_if_else(benchmark::State& state, bool nulls)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto input_type  = cudf::type_to_id<TypeParam>();
  auto bool_type   = cudf::type_id::BOOL8;
  auto const input = create_random_table({input_type, input_type, bool_type}, row_count{n_rows});

  if (!nulls) {
    input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
  }

  cudf::column_view decision(input->view().column(2));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view lhs(input->view().column(0));

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    cudf::copy_if_else(lhs, rhs, decision);
  }

  auto const bytes_read    = n_rows * (sizeof(TypeParam) + sizeof(bool));
  auto const bytes_written = n_rows * sizeof(TypeParam);
  auto const null_bytes    = nulls ? 2 * cudf::bitmask_allocation_size_bytes(n_rows) : 0;

  // Use number of bytes read and written.
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (bytes_read + bytes_written + null_bytes));
}

#define COPY_BENCHMARK_DEFINE(name, type, b)                  \
  BENCHMARK_DEFINE_F(CopyIfElse, name)                        \
  (::benchmark::State & st) { BM_copy_if_else<type>(st, b); } \
  BENCHMARK_REGISTER_F(CopyIfElse, name)                      \
    ->RangeMultiplier(8)                                      \
    ->Ranges({{1 << 12, 1 << 27}})                            \
    ->UseManualTime()                                         \
    ->Unit(benchmark::kMillisecond);

COPY_BENCHMARK_DEFINE(int16, int16_t, true)
COPY_BENCHMARK_DEFINE(uint32, uint32_t, true)
COPY_BENCHMARK_DEFINE(float64, double, true)
COPY_BENCHMARK_DEFINE(int16_no_nulls, int16_t, false)
COPY_BENCHMARK_DEFINE(uint32_no_nulls, uint32_t, false)
COPY_BENCHMARK_DEFINE(float64_no_nulls, double, false)
