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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cmath>

namespace cudf {

class ScatterLists : public cudf::benchmark {
};

template <class TypeParam, bool coalesce>
void BM_lists_scatter(::benchmark::State& state)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = rmm::mr::get_current_device_resource();

  const size_type base_size{(size_type)state.range(0)};
  const size_type num_elements_per_row{(size_type)state.range(1)};
  const size_type num_rows = (size_type)ceil(double(base_size) / num_elements_per_row);

  auto source_base_col = make_fixed_width_column(
    data_type{type_to_id<TypeParam>()}, base_size, mask_state::UNALLOCATED, stream, mr);
  auto target_base_col = make_fixed_width_column(
    data_type{type_to_id<TypeParam>()}, base_size, mask_state::UNALLOCATED, stream, mr);
  thrust::sequence(rmm::exec_policy(stream),
                   source_base_col->mutable_view().begin<TypeParam>(),
                   source_base_col->mutable_view().end<TypeParam>());
  thrust::sequence(rmm::exec_policy(stream),
                   target_base_col->mutable_view().begin<TypeParam>(),
                   target_base_col->mutable_view().end<TypeParam>());

  auto source_offsets = make_fixed_width_column(
    data_type{type_to_id<offset_type>()}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto target_offsets = make_fixed_width_column(
    data_type{type_to_id<offset_type>()}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  thrust::sequence(rmm::exec_policy(stream),
                   source_offsets->mutable_view().begin<offset_type>(),
                   source_offsets->mutable_view().end<offset_type>(),
                   0,
                   num_elements_per_row);
  thrust::sequence(rmm::exec_policy(stream),
                   target_offsets->mutable_view().begin<offset_type>(),
                   target_offsets->mutable_view().end<offset_type>(),
                   0,
                   num_elements_per_row);

  auto source = make_lists_column(num_rows,
                                  std::move(source_offsets),
                                  std::move(source_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, mask_state::UNALLOCATED),
                                  stream,
                                  mr);
  auto target = make_lists_column(num_rows,
                                  std::move(target_offsets),
                                  std::move(target_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, mask_state::UNALLOCATED),
                                  stream,
                                  mr);

  auto scatter_map = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto m_scatter_map = scatter_map->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   m_scatter_map.begin<size_type>(),
                   m_scatter_map.end<size_type>(),
                   num_rows - 1,
                   -1);

  if (not coalesce) {
    thrust::default_random_engine g;
    thrust::shuffle(rmm::exec_policy(stream),
                    m_scatter_map.begin<size_type>(),
                    m_scatter_map.begin<size_type>(),
                    g);
  }

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    scatter(table_view{{*source}}, *scatter_map, table_view{{*target}}, false, mr);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * 2 *
                          sizeof(TypeParam));
}

#define SBM_BENCHMARK_DEFINE(name, type, coalesce)                                \
  BENCHMARK_DEFINE_F(ScatterLists, name)(::benchmark::State & state)              \
  {                                                                               \
    BM_lists_scatter<type, coalesce>(state);                                      \
  }                                                                               \
  BENCHMARK_REGISTER_F(ScatterLists, name)                                        \
    ->RangeMultiplier(8)                                                          \
    ->Ranges({{1 << 10, 1 << 25}, {64, 2048}}) /* 1K-1B rows, 64-2048 elements */ \
    ->UseManualTime();

SBM_BENCHMARK_DEFINE(double_type_colesce_o, double, true);
SBM_BENCHMARK_DEFINE(double_type_colesce_x, double, false);

}  // namespace cudf
