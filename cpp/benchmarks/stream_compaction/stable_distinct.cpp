/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

template <typename Type>
void nvbench_stable_distinct(nvbench::state& state, nvbench::type_list<Type>)
{
  cudf::size_type const num_rows = state.get_int64("NumRows");

  data_profile profile = data_profile_builder().cardinality(0).null_probability(0.01).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);

  auto source_column = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);

  auto input_column = source_column->view();
  auto input_table  = cudf::table_view({input_column, input_column, input_column, input_column});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::stable_distinct(input_table,
                                        {0},
                                        cudf::duplicate_keep_option::KEEP_ANY,
                                        cudf::null_equality::EQUAL,
                                        cudf::nan_equality::ALL_EQUAL);
  });
}

using data_type = nvbench::type_list<bool, int8_t, int32_t, int64_t, float, cudf::timestamp_ms>;

NVBENCH_BENCH_TYPES(nvbench_stable_distinct, NVBENCH_TYPE_AXES(data_type))
  .set_name("stable_distinct")
  .set_type_axes_names({"Type"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000});

template <typename Type>
void nvbench_stable_distinct_list(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const size               = state.get_int64("ColumnSize");
  auto const dtype              = cudf::type_to_id<Type>();
  double const null_probability = state.get_float64("null_probability");

  auto builder = data_profile_builder().null_probability(null_probability);
  if (dtype == cudf::type_id::LIST) {
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 4)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 4)
      .list_depth(1);
  } else {
    // We're comparing stable_distinct() on a non-nested column to that on a list column with the
    // same number of stable_distinct rows. The max list size is 4 and the number of distinct values
    // in the list's child is 5. So the number of distinct rows in the list = 1 + 5 + 5^2 + 5^3 +
    // 5^4 = 781 We want this column to also have 781 distinct values.
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 781);
  }

  auto const table = create_random_table(
    {dtype}, table_size_bytes{static_cast<size_t>(size)}, data_profile{builder}, 0);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::stable_distinct(*table,
                                        {0},
                                        cudf::duplicate_keep_option::KEEP_ANY,
                                        cudf::null_equality::EQUAL,
                                        cudf::nan_equality::ALL_EQUAL);
  });
}

NVBENCH_BENCH_TYPES(nvbench_stable_distinct_list,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, cudf::list_view>))
  .set_name("stable_distinct_list")
  .set_type_axes_names({"Type"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("ColumnSize", {100'000'000});
