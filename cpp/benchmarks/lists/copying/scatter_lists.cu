/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cmath>

class ScatterLists : public cudf::benchmark {};

template <class TypeParam, bool coalesce>
void BM_lists_scatter(::benchmark::State& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  cudf::size_type const base_size{(cudf::size_type)state.range(0)};
  cudf::size_type const num_elements_per_row{(cudf::size_type)state.range(1)};
  auto const num_rows = (cudf::size_type)ceil(double(base_size) / num_elements_per_row);

  auto source_base_col = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                                 base_size,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  auto target_base_col = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                                 base_size,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  thrust::sequence(rmm::exec_policy(stream),
                   source_base_col->mutable_view().begin<TypeParam>(),
                   source_base_col->mutable_view().end<TypeParam>());
  thrust::sequence(rmm::exec_policy(stream),
                   target_base_col->mutable_view().begin<TypeParam>(),
                   target_base_col->mutable_view().end<TypeParam>());

  auto source_offsets =
    make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                            num_rows + 1,
                            cudf::mask_state::UNALLOCATED,
                            stream,
                            mr);
  auto target_offsets =
    make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                            num_rows + 1,
                            cudf::mask_state::UNALLOCATED,
                            stream,
                            mr);

  thrust::sequence(rmm::exec_policy(stream),
                   source_offsets->mutable_view().begin<cudf::size_type>(),
                   source_offsets->mutable_view().end<cudf::size_type>(),
                   0,
                   num_elements_per_row);
  thrust::sequence(rmm::exec_policy(stream),
                   target_offsets->mutable_view().begin<cudf::size_type>(),
                   target_offsets->mutable_view().end<cudf::size_type>(),
                   0,
                   num_elements_per_row);

  auto source = make_lists_column(num_rows,
                                  std::move(source_offsets),
                                  std::move(source_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, cudf::mask_state::UNALLOCATED),
                                  stream,
                                  mr);
  auto target = make_lists_column(num_rows,
                                  std::move(target_offsets),
                                  std::move(target_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, cudf::mask_state::UNALLOCATED),
                                  stream,
                                  mr);

  auto scatter_map   = make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                             num_rows,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
  auto m_scatter_map = scatter_map->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   m_scatter_map.begin<cudf::size_type>(),
                   m_scatter_map.end<cudf::size_type>(),
                   num_rows - 1,
                   -1);

  if (not coalesce) {
    thrust::default_random_engine g;
    thrust::shuffle(rmm::exec_policy(stream),
                    m_scatter_map.begin<cudf::size_type>(),
                    m_scatter_map.begin<cudf::size_type>(),
                    g);
  }

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    scatter(cudf::table_view{{*source}},
            *scatter_map,
            cudf::table_view{{*target}},
            cudf::get_default_stream(),
            mr);
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

SBM_BENCHMARK_DEFINE(double_coalesced, double, true);
SBM_BENCHMARK_DEFINE(double_shuffled, double, false);
