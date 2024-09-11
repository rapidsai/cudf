/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda/functional>

#include <cudf/detail/tdigest/tdigest.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

class TDigest : public cudf::benchmark {};

static void BM_tdigest_merge(benchmark::State& state)
{
  cudf::size_type const num_tdigests{(cudf::size_type)state.range(0)};
  cudf::size_type const tdigest_size{(cudf::size_type)state.range(1)};
  cudf::size_type const tdigests_per_group{(cudf::size_type)state.range(2)};
  cudf::size_type const max_centroids{(cudf::size_type)state.range(3)};
  auto const num_groups = num_tdigests / tdigests_per_group;
  auto const total_centroids = num_tdigests * tdigest_size;

  auto stream = cudf::get_default_stream();
  auto mr = rmm::mr::get_current_device_resource();

  constexpr int base_value = 5;
  
  // construct inner means/weights
  auto val_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<double>([tdigest_size] (cudf::size_type i) {
    return static_cast<double>(base_value + i % tdigest_size);
  }));
  auto one_iter = thrust::make_constant_iterator(1);
  cudf::test::fixed_width_column_wrapper<double> means(val_iter, val_iter + total_centroids);
  cudf::test::fixed_width_column_wrapper<double> weights(one_iter, one_iter + total_centroids);
  std::vector<std::unique_ptr<cudf::column>> inner_struct_children;
  inner_struct_children.push_back(means.release());
  inner_struct_children.push_back(weights.release());
  cudf::test::structs_column_wrapper inner_struct(std::move(inner_struct_children));

  // construct the tdigest lists themselves
  auto offset_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<cudf::size_type>([tdigest_size] (cudf::size_type i){
    return i * tdigest_size;
  }));  
  cudf::test::fixed_width_column_wrapper<int> offsets(offset_iter, offset_iter + num_tdigests + 1);
  auto list_col = cudf::make_lists_column(num_tdigests,
                                          offsets.release(),
                                          inner_struct.release(),
                                          0,
                                          {},
                                          stream,
                                          mr);

  // min and max columns
  auto min_iter = thrust::make_constant_iterator(base_value);
  auto max_iter = thrust::make_constant_iterator(base_value + (tdigest_size - 1));
  cudf::test::fixed_width_column_wrapper<double> mins(min_iter, min_iter + num_tdigests);
  cudf::test::fixed_width_column_wrapper<double> maxes(max_iter, max_iter + num_tdigests);

  // assemble the whole thing
  std::vector<std::unique_ptr<cudf::column>> tdigest_children;
  tdigest_children.push_back(std::move(list_col));
  tdigest_children.push_back(mins.release());
  tdigest_children.push_back(maxes.release());
  cudf::test::structs_column_wrapper tdigest(std::move(tdigest_children));

  rmm::device_uvector<cudf::size_type> group_offsets(num_groups+1, stream, mr);
  rmm::device_uvector<cudf::size_type> group_labels(num_tdigests, stream, mr);
  auto group_offset_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<cudf::size_type>([tdigests_per_group] __device__ (cudf::size_type i){
    return i * tdigests_per_group;
  }));
  thrust::copy(rmm::exec_policy_nosync(stream, mr),
               group_offset_iter,
               group_offset_iter + num_groups + 1,
               group_offsets.begin());
  auto group_label_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<cudf::size_type>([tdigests_per_group] __device__ (cudf::size_type i){
    return i / tdigests_per_group;
  }));
  thrust::copy(rmm::exec_policy_nosync(stream, mr),
               group_label_iter,
               group_label_iter + num_tdigests,
               group_labels.begin());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, stream);

    auto result = cudf::tdigest::detail::group_merge_tdigest(tdigest,
                                                             group_offsets,
                                                             group_labels,
                                                             num_groups,
                                                             max_centroids,
                                                             stream,
                                                             mr);
  }
}

#define TDIGEST_BENCHMARK_DEFINE(name, num_tdigests, tdigest_size, tdigests_per_group, max_centroids)   \
  BENCHMARK_DEFINE_F(TDigest, name)                                                                     \
  (::benchmark::State & st) { BM_tdigest_merge(st); }                                                   \
  BENCHMARK_REGISTER_F(TDigest, name)                                                                   \
    ->Args({num_tdigests, tdigest_size, tdigests_per_group, max_centroids})                             \
    ->Unit(benchmark::kMillisecond)                                                                     \
    ->UseManualTime()                                                                                   \
    ->Iterations(8)

TDIGEST_BENCHMARK_DEFINE(many_tiny_groups, 1'000'000, 1, 1, 10000);
TDIGEST_BENCHMARK_DEFINE(many_tiny_groups2, 1'000'000, 1, 1, 1000);

TDIGEST_BENCHMARK_DEFINE(many_small_groups, 3'000'000, 3, 3, 10000);
TDIGEST_BENCHMARK_DEFINE(many_small_groups2, 3'000'000, 3, 3, 1000);