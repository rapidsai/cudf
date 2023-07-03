/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  T value                             = 0,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto s = new ScalarType(value, true, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T>
struct value_func {
  T* data;
  cudf::size_type offset;

  __device__ T operator()(int idx) { return data[idx - offset]; }
};

struct validity_func {
  cudf::size_type size;
  cudf::size_type offset;

  __device__ bool operator()(int idx)
  {
    auto source_idx = idx - offset;
    return source_idx < 0 || source_idx >= size;
  }
};

template <bool use_validity, int shift_factor>
static void BM_shift(benchmark::State& state)
{
  cudf::size_type size   = state.range(0);
  cudf::size_type offset = size * (static_cast<double>(shift_factor) / 100.0);

  auto const input_table =
    create_sequence_table({cudf::type_to_id<int>()},
                          row_count{size},
                          use_validity ? std::optional<double>{1.0} : std::nullopt);
  cudf::column_view input{input_table->get_column(0)};

  auto fill = use_validity ? make_scalar<int>() : make_scalar<int>(777);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::shift(input, offset, *fill);
  }
}

class Shift : public cudf::benchmark {};

#define SHIFT_BM_BENCHMARK_DEFINE(name, use_validity, shift_factor) \
  BENCHMARK_DEFINE_F(Shift, name)(::benchmark::State & state)       \
  {                                                                 \
    BM_shift<use_validity, shift_factor>(state);                    \
  }                                                                 \
  BENCHMARK_REGISTER_F(Shift, name)                                 \
    ->RangeMultiplier(32)                                           \
    ->Range(1 << 10, 1 << 30)                                       \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

SHIFT_BM_BENCHMARK_DEFINE(shift_zero, false, 0);
SHIFT_BM_BENCHMARK_DEFINE(shift_zero_nullable_out, true, 0);

SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent, false, 10);
SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent_nullable_out, true, 10);

SHIFT_BM_BENCHMARK_DEFINE(shift_half, false, 50);
SHIFT_BM_BENCHMARK_DEFINE(shift_half_nullable_out, true, 50);

SHIFT_BM_BENCHMARK_DEFINE(shift_full, false, 100);
SHIFT_BM_BENCHMARK_DEFINE(shift_full_nullable_out, true, 100);
