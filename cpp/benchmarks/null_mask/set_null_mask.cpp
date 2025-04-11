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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/null_mask.hpp>

class SetNullmask : public cudf::benchmark {};

void BM_setnullmask(benchmark::State& state)
{
  cudf::size_type const size{(cudf::size_type)state.range(0)};
  rmm::device_buffer mask = cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED);
  auto begin = 0, end = size;

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), begin, end, true);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size / 8);
}

void BM_setnullmask_bulk(benchmark::State& state)
{
  cudf::size_type const size{(cudf::size_type)state.range(0)};
  cudf::size_type const num_masks{(cudf::size_type)state.range(1)};

  std::vector<cudf::size_type> begin_bits(num_masks, 0);
  std::vector<cudf::size_type> end_bits(num_masks, size);
  auto valids = thrust::host_vector<bool>(num_masks, true);

  std::vector<rmm::device_buffer> masks(num_masks);
  std::vector<cudf::bitmask_type*> masks_ptr(num_masks);
  for (cudf::size_type i = 0; i < num_masks; ++i) {
    masks[i]     = cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    masks_ptr[i] = static_cast<cudf::bitmask_type*>(masks[i].data());
  }

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::set_null_masks_bulk(masks_ptr, begin_bits, end_bits, valids);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * num_masks / 8);
}

#define NBM_BENCHMARK_DEFINE(name)                                                             \
  BENCHMARK_DEFINE_F(SetNullmask, name)(::benchmark::State & state) { BM_setnullmask(state); } \
  BENCHMARK_REGISTER_F(SetNullmask, name)                                                      \
    ->RangeMultiplier(1 << 10)                                                                 \
    ->Range(1 << 10, 1 << 30)                                                                  \
    ->UseManualTime();

#define NBM_BENCHMARK_DEFINE_BULK(name)                                     \
  BENCHMARK_DEFINE_F(SetNullmaskBulk, name)(::benchmark::State & state)     \
  {                                                                         \
    BM_setnullmask_bulk(state);                                             \
  }                                                                         \
  BENCHMARK_REGISTER_F(SetNullmaskBulk, name)                               \
    ->ArgsProduct({                                                         \
      benchmark::CreateRange(1 << 5, 1 << 20, 1 << 5), /* Powers of 1024 */ \
      benchmark::CreateRange(1 << 2, 1 << 8, 1 << 2)   /* Powers of 2 */    \
    })
-> UseManualTime();

NBM_BENCHMARK_DEFINE(SetNullMaskKernel);
NBM_BENCHMARK_DEFINE_BULK(SetNullMaskBulkKernel);