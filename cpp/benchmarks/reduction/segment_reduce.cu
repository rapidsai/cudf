/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <nvbench/nvbench.cuh>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/device_vector.h>

#include <memory>
#include <type_traits>

namespace cudf {

bool constexpr is_boolean_output_agg(segmented_reduce_aggregation::Kind kind)
{
  return kind == segmented_reduce_aggregation::ALL || kind == segmented_reduce_aggregation::ANY;
}

template <segmented_reduce_aggregation::Kind kind>
std::unique_ptr<segmented_reduce_aggregation> make_simple_aggregation()
{
  switch (kind) {
    case segmented_reduce_aggregation::SUM:
      return make_sum_aggregation<segmented_reduce_aggregation>();
    case segmented_reduce_aggregation::PRODUCT:
      return make_product_aggregation<segmented_reduce_aggregation>();
    case segmented_reduce_aggregation::MIN:
      return make_min_aggregation<segmented_reduce_aggregation>();
    case segmented_reduce_aggregation::MAX:
      return make_max_aggregation<segmented_reduce_aggregation>();
    case segmented_reduce_aggregation::ALL:
      return make_all_aggregation<segmented_reduce_aggregation>();
    case segmented_reduce_aggregation::ANY:
      return make_any_aggregation<segmented_reduce_aggregation>();
    default: CUDF_FAIL("Unsupported simple segmented aggregation");
  }
}

template <typename InputType>
std::pair<std::unique_ptr<column>, thrust::device_vector<size_type>> make_test_data(
  nvbench::state& state)
{
  auto const column_size{size_type(state.get_int64("column_size"))};
  auto const num_segments{size_type(state.get_int64("num_segments"))};

  auto segment_length = column_size / num_segments;

  auto const dtype     = cudf::type_to_id<InputType>();
  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    dtype, distribution_id::UNIFORM, 0, 100);
  auto input = create_random_column(dtype, row_count{column_size}, profile);

  auto offset_it =
    detail::make_counting_transform_iterator(0, [column_size, segment_length] __device__(auto i) {
      return column_size < i * segment_length ? column_size : i * segment_length;
    });

  thrust::device_vector<size_type> d_offsets(offset_it, offset_it + num_segments + 1);

  return std::pair(std::move(input), d_offsets);
}

template <typename InputType, typename OutputType, aggregation::Kind kind>
std::enable_if_t<!is_boolean_output_agg(kind) || std::is_same_v<OutputType, bool>, void>
BM_Simple_Segmented_Reduction(nvbench::state& state,
                              nvbench::type_list<InputType, OutputType, nvbench::enum_type<kind>>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const column_size{size_type(state.get_int64("column_size"))};
  auto [input, offsets] = make_test_data<InputType>(state);
  auto agg              = make_simple_aggregation<kind>();

  state.add_element_count(column_size);
  state.add_global_memory_reads<InputType>(column_size);
  state.add_global_memory_writes<OutputType>(column_size);

  state.exec(
    nvbench::exec_tag::sync,
    [input_view = input->view(), offset_span = device_span<size_type>{offsets}, &agg](
      nvbench::launch& launch) {
      segmented_reduce(
        input_view, offset_span, *agg, data_type{type_to_id<OutputType>()}, null_policy::INCLUDE);
    });
}

template <typename InputType, typename OutputType, aggregation::Kind kind>
std::enable_if_t<is_boolean_output_agg(kind) && !std::is_same_v<OutputType, bool>, void>
BM_Simple_Segmented_Reduction(nvbench::state& state,
                              nvbench::type_list<InputType, OutputType, nvbench::enum_type<kind>>)
{
  state.skip("Invalid combination of dtype and aggregation type.");
}

using Types = nvbench::type_list<bool, int32_t, float, double>;
// Skip benchmarking MAX/ANY since they are covered by MIN/ALL respectively.
using AggKinds = nvbench::
  enum_type_list<aggregation::SUM, aggregation::PRODUCT, aggregation::MIN, aggregation::ALL>;

NVBENCH_BENCH_TYPES(BM_Simple_Segmented_Reduction, NVBENCH_TYPE_AXES(Types, Types, AggKinds))
  .set_name("segmented_reduction_simple")
  .set_type_axes_names({"InputType", "OutputType", "AggregationKinds"})
  .add_int64_axis("column_size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_segments", {1'000, 10'000, 100'000});

}  // namespace cudf
