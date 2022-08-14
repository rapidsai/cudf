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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

void bench_groupby_struct_keys(nvbench::state& state)
{
  cudf::rmm_pool_raii pool_raii;

  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{static_cast<cudf::size_type>(state.get_int64("NumRows"))};
  const cudf::size_type n_cols{1};
  const cudf::size_type depth{static_cast<cudf::size_type>(state.get_int64("Depth"))};
  const bool nulls{static_cast<bool>(state.get_int64("Nulls"))};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&]() {
    auto const elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    if (!nulls) return column_wrapper(elements, elements + n_rows);
    auto valids =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 10 != 0; });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  std::vector<std::unique_ptr<cudf::column>> child_cols = std::move(cols);
  // Add some layers
  for (int i = 0; i < depth; i++) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 100 * (i + 1));
    std::generate_n(
      std::back_inserter(struct_validity), n_rows, [&]() { return bool_distribution(generator); });
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols), struct_validity);
    child_cols = std::vector<std::unique_ptr<cudf::column>>{};
    child_cols.push_back(struct_col.release());
  }
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);

  auto const keys_table = cudf::table(std::move(child_cols));
  auto const vals_table =
    create_random_table({cudf::type_to_id<int64_t>()}, row_count{n_rows}, profile);

  cudf::groupby::groupby gb_obj(keys_table.view());

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals_table->get_column(0).view();
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  // Set up nvbench default stream
  auto stream = cudf::default_stream_value;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto const result = gb_obj.aggregate(requests); });
}

NVBENCH_BENCH(bench_groupby_struct_keys)
  .set_name("groupby_struct_keys")
  .add_int64_power_of_two_axis("NumRows", {10, 16, 20})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
