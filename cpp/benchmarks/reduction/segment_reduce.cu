/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

bool constexpr is_boolean_output_agg(cudf::segmented_reduce_aggregation::Kind kind)
{
  return kind == cudf::segmented_reduce_aggregation::ALL ||
         kind == cudf::segmented_reduce_aggregation::ANY;
}

template <cudf::segmented_reduce_aggregation::Kind kind>
std::unique_ptr<cudf::segmented_reduce_aggregation> make_simple_aggregation()
{
  switch (kind) {
    case cudf::segmented_reduce_aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::MAX:
      return cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::ALL:
      return cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::ANY:
      return cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
    default: CUDF_FAIL("Unsupported simple segmented aggregation");
  }
}

template <typename DataType>
std::pair<std::unique_ptr<cudf::column>, thrust::device_vector<cudf::size_type>> make_test_data(
  nvbench::state& state)
{
  auto const column_size{cudf::size_type(state.get_int64("column_size"))};
  auto const num_segments{cudf::size_type(state.get_int64("num_segments"))};

  auto segment_length = column_size / num_segments;

  auto const dtype     = cudf::type_to_id<DataType>();
  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    dtype, distribution_id::UNIFORM, 0, 100);
  auto input = create_random_column(dtype, row_count{column_size}, profile);

  auto offset_it = cudf::detail::make_counting_transform_iterator(
    0, [column_size, segment_length] __device__(auto i) {
      return column_size < i * segment_length ? column_size : i * segment_length;
    });

  thrust::device_vector<cudf::size_type> d_offsets(offset_it, offset_it + num_segments + 1);

  return std::pair(std::move(input), d_offsets);
}

template <typename DataType, cudf::aggregation::Kind kind>
void BM_Simple_Segmented_Reduction(nvbench::state& state,
                                   nvbench::type_list<DataType, nvbench::enum_type<kind>>)
{
  auto const column_size{cudf::size_type(state.get_int64("column_size"))};
  auto const num_segments{cudf::size_type(state.get_int64("num_segments"))};

  auto [input, offsets] = make_test_data<DataType>(state);
  auto agg              = make_simple_aggregation<kind>();

  auto output_type = is_boolean_output_agg(kind) ? cudf::data_type{cudf::type_id::BOOL8}
                                                 : cudf::data_type{cudf::type_to_id<DataType>()};

  state.add_element_count(column_size);
  state.add_global_memory_reads<DataType>(column_size);
  if (is_boolean_output_agg(kind)) {
    state.add_global_memory_writes<nvbench::int8_t>(num_segments);  // BOOL8
  } else {
    state.add_global_memory_writes<DataType>(num_segments);
  }

  auto const input_view  = input->view();
  auto const offset_span = cudf::device_span<cudf::size_type>{offsets};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync, [input_view, output_type, offset_span, &agg](nvbench::launch& launch) {
      segmented_reduce(input_view, offset_span, *agg, output_type, cudf::null_policy::INCLUDE);
    });
}

using Types = nvbench::type_list<bool, int32_t, float, double>;
// Skip benchmarking MAX/ANY since they are covered by MIN/ALL respectively.
using AggKinds = nvbench::enum_type_list<cudf::aggregation::SUM,
                                         cudf::aggregation::PRODUCT,
                                         cudf::aggregation::MIN,
                                         cudf::aggregation::ALL>;

NVBENCH_BENCH_TYPES(BM_Simple_Segmented_Reduction, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("segmented_reduction_simple")
  .set_type_axes_names({"DataType", "AggregationKinds"})
  .add_int64_axis("column_size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_segments", {1'000, 10'000, 100'000});
