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

#include <benchmarks/common/generate_nested_types.hpp>

#include <cudf/detail/sorting.hpp>

#include <nvbench/nvbench.cuh>

namespace {
constexpr cudf::size_type min_val = 0;
constexpr cudf::size_type max_val = 100;

void sort_multiple_lists(nvbench::state& state)
{
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const input_table = create_lists_data(state, num_columns, min_val, max_val);
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::detail::sorted_order(
      *input_table, {}, {}, stream, rmm::mr::get_current_device_resource());
  });
}

void sort_lists_of_structs(nvbench::state& state)
{
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const lists_table = create_lists_data(state, num_columns, min_val, max_val);

  // After having a table of (multiple) lists columns, convert those lists columns into lists of
  // structs columns. The children of these structs columns are also children of the original lists
  // columns.
  // Such resulted lists-of-structs columns are very similar to the original lists-of-integers
  // columns so their benchmarks can be somewhat comparable.
  std::vector<cudf::column_view> lists_of_structs;
  for (auto const& col : lists_table->view()) {
    auto const child = col.child(cudf::lists_column_view::child_column_index);

    // Put the child column under a struct column having the same null mask/null count.
    auto const new_child = cudf::column_view{cudf::data_type{cudf::type_id::STRUCT},
                                             child.size(),
                                             nullptr,
                                             child.null_mask(),
                                             child.null_count(),
                                             child.offset(),
                                             {child}};
    auto const converted_col =
      cudf::column_view{cudf::data_type{cudf::type_id::LIST},
                        col.size(),
                        nullptr,
                        col.null_mask(),
                        col.null_count(),
                        col.offset(),
                        {col.child(cudf::lists_column_view::offsets_column_index), new_child}};
    lists_of_structs.push_back(converted_col);
  }

  auto const input_table = cudf::table_view{lists_of_structs};
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    cudf::detail::sorted_order(input_table, {}, {}, stream, rmm::mr::get_current_device_resource());
  });
}

}  // namespace

void nvbench_sort_lists(nvbench::state& state)
{
  auto const has_lists_of_structs = state.get_int64("lists_of_structs") > 0;
  if (has_lists_of_structs) {
    sort_lists_of_structs(state);
  } else {
    sort_multiple_lists(state);
  }
}

NVBENCH_BENCH(nvbench_sort_lists)
  .set_name("sort_list")
  .add_int64_power_of_two_axis("size_bytes", {10, 18, 24, 28})
  .add_int64_axis("depth", {1, 4})
  .add_int64_axis("num_columns", {1})
  .add_int64_axis("lists_of_structs", {0, 1})
  .add_float64_axis("null_frequency", {0, 0.2});
