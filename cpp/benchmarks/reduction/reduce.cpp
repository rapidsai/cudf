/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <benchmarks/common/benchmark_utilities.hpp>
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>

template <cudf::reduce_aggregation::Kind kind>
static std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation()
{
  switch (kind) {
    case cudf::reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MEAN:
      return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::STD:
      return cudf::make_std_aggregation<cudf::reduce_aggregation>();
    default: CUDF_FAIL("Unsupported reduce aggregation in this benchmark");
  }
}

template <typename DataType, cudf::reduce_aggregation::Kind kind>
static void reduction(nvbench::state& state, nvbench::type_list<DataType, nvbench::enum_type<kind>>)
{
  auto const size = static_cast<cudf::size_type>(state.get_int64("size"));
  if (cudf::is_chrono<DataType>() && kind != cudf::aggregation::MIN) {
    state.skip("Skip chrono types for some aggregations");
  }

  auto const input_type = cudf::type_to_id<DataType>();
  data_profile const profile =
    data_profile_builder().no_validity().distribution(input_type, distribution_id::UNIFORM, 0, 100);
  auto const input_column = create_random_column(input_type, row_count{size}, profile);

  cudf::data_type output_type =
    (kind == cudf::aggregation::MEAN || kind == cudf::aggregation::VARIANCE ||
     kind == cudf::aggregation::STD)
      ? cudf::data_type{cudf::type_id::FLOAT64}
      : input_column->type();

  auto agg = make_reduce_aggregation<kind>();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<DataType>(1);

  state.exec(nvbench::exec_tag::sync, [&input_column, output_type, &agg](nvbench::launch& launch) {
    cudf::reduce(*input_column, *agg, output_type);
  });

  set_throughputs(state);
}

using Types    = nvbench::type_list<int32_t, int64_t, double, cudf::timestamp_ms>;
using AggKinds = nvbench::enum_type_list<cudf::reduce_aggregation::MIN,
                                         cudf::reduce_aggregation::SUM,
                                         cudf::reduce_aggregation::PRODUCT,
                                         cudf::reduce_aggregation::VARIANCE,
                                         cudf::reduce_aggregation::STD,
                                         cudf::reduce_aggregation::MEAN>;

NVBENCH_BENCH_TYPES(reduction, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("reduction")
  .set_type_axes_names({"DataType", "AggKinds"})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000});
