/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transpose.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

static void BM_transpose(benchmark::State& state)
{
  auto count                    = state.range(0);
  constexpr auto column_type_id = cudf::type_id::INT32;
  auto int_column_generator =
    thrust::make_transform_iterator(thrust::counting_iterator(0), [count](int i) {
      return cudf::make_numeric_column(
        cudf::data_type{column_type_id}, count, cudf::mask_state::ALL_VALID);
    });

  auto input_table = cudf::table(std::vector(int_column_generator, int_column_generator + count));
  auto input       = input_table.view();

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::transpose(input);
  }

  // Collect memory statistics.
  auto const bytes_read = static_cast<uint64_t>(input.num_columns()) * input.num_rows() *
                          sizeof(cudf::id_to_type<column_type_id>);
  auto const bytes_written = bytes_read;
  // Account for nullability in input and output.
  auto const null_bytes = 2 * static_cast<uint64_t>(input.num_columns()) *
                          cudf::bitmask_allocation_size_bytes(input.num_rows());

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (bytes_read + bytes_written + null_bytes));
}

class Transpose : public cudf::benchmark {};

#define TRANSPOSE_BM_BENCHMARK_DEFINE(name)                                                \
  BENCHMARK_DEFINE_F(Transpose, name)(::benchmark::State & state) { BM_transpose(state); } \
  BENCHMARK_REGISTER_F(Transpose, name)                                                    \
    ->RangeMultiplier(4)                                                                   \
    ->Range(4, 4 << 13)                                                                    \
    ->UseManualTime()                                                                      \
    ->Unit(benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(transpose_simple);
