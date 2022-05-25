/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

//==================================================================================================
auto create_offsets(cudf::size_type n_groups, rmm::cuda_stream_view stream)
{
  // This is the maximum size of each group.
  constexpr cudf::size_type max_int = 1000;

  auto table_profile = data_profile{};
  table_profile.set_distribution_params(cudf::type_id::INT32, distribution_id::UNIFORM, 0, max_int);
  auto sizes =
    std::move(create_random_table({cudf::type_id::INT32}, row_count{n_groups}, table_profile)
                ->release()
                .front());
  auto const sizes_view = sizes->mutable_view();

  thrust::exclusive_scan(rmm::exec_policy(),
                         sizes_view.template begin<cudf::size_type>(),
                         sizes_view.template end<cudf::size_type>(),
                         sizes_view.template begin<cudf::size_type>());

  cudf::size_type n_elements;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&n_elements,
                                sizes_view.template end<cudf::size_type>() - 1,
                                sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  return std::pair{std::move(sizes), n_elements};
}

//==================================================================================================
template <typename InputIterator, typename OutputIterator>
void old_way(InputIterator offsets_begin,
             InputIterator offsets_end,
             OutputIterator out_begin,
             OutputIterator out_end,
             rmm::cuda_stream_view stream)
{
  thrust::uninitialized_fill(rmm::exec_policy(stream), out_begin, out_end, cudf::size_type{0});
  thrust::scatter(
    rmm::exec_policy(stream),
    thrust::make_constant_iterator(1, 1),
    thrust::make_constant_iterator(
      1, static_cast<cudf::size_type>(thrust::distance(offsets_begin, offsets_end)) - 1),
    offsets_begin + 1,
    out_begin);
  thrust::inclusive_scan(rmm::exec_policy(stream), out_begin, out_end, out_begin);
}

//==================================================================================================
template <typename InputIterator, typename OutputIterator>
void new_way(InputIterator offsets_begin,
             InputIterator offsets_end,
             OutputIterator out_begin,
             OutputIterator out_end,
             rmm::cuda_stream_view stream)
{
  auto const zero_normalized_offsets = thrust::make_transform_iterator(
    offsets_begin, [offsets_begin] __device__(auto const idx) { return idx - *offsets_begin; });

  // The output labels from `upper_bound` will start from `1`.
  // This will shift the result values back to start from `0`.
  using OutputType  = typename thrust::iterator_value<OutputIterator>::type;
  auto const output = thrust::make_transform_output_iterator(
    out_begin, [] __device__(auto const idx) { return idx - OutputType{1}; });

  thrust::upper_bound(rmm::exec_policy(stream),
                      zero_normalized_offsets,
                      zero_normalized_offsets + thrust::distance(offsets_begin, offsets_end),
                      thrust::make_counting_iterator<OutputType>(0),
                      thrust::make_counting_iterator<OutputType>(
                        static_cast<OutputType>(thrust::distance(out_begin, out_end))),
                      output);
}

//==================================================================================================
template <bool use_old>
void BM_labeling(benchmark::State& state)
{
  auto const n_groups = static_cast<cudf::size_type>(state.range(0));
  auto const stream   = rmm::cuda_stream_default;

  auto const [offsets, n_labels] = create_offsets(n_groups, stream);
  auto const offsets_view        = offsets->view();
  auto labels                    = rmm::device_uvector<cudf::size_type>(n_labels, stream);

  for (auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    if constexpr (use_old) {
      old_way(offsets_view.template begin<cudf::size_type>(),
              offsets_view.template end<cudf::size_type>(),
              labels.begin(),
              labels.end(),
              stream);
    } else {
      new_way(offsets_view.template begin<cudf::size_type>(),
              offsets_view.template end<cudf::size_type>(),
              labels.begin(),
              labels.end(),
              stream);
    }
  }
}

//==================================================================================================
class Labeling : public cudf::benchmark {
};

#define MIN_RANGE 1'000
#define MAX_RANGE 4'200'000

#define REGISTER_BENCHMARK(name, use_old)                                                         \
  BENCHMARK_DEFINE_F(Labeling, name)(::benchmark::State & state) { BM_labeling<use_old>(state); } \
  BENCHMARK_REGISTER_F(Labeling, name)                                                            \
    ->UseManualTime()                                                                             \
    ->Unit(benchmark::kMillisecond)                                                               \
    ->RangeMultiplier(4)                                                                          \
    ->Ranges({{MIN_RANGE, MAX_RANGE}});

REGISTER_BENCHMARK(LabelingOldWay, true)
// REGISTER_BENCHMARK(LabelingNewWay, false)
