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

#pragma once

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

inline std::unique_ptr<cudf::table> create_lists_data(nvbench::state& state)
{
  const size_t size_bytes(state.get_int64("size_bytes"));
  const cudf::size_type depth{static_cast<cudf::size_type>(state.get_int64("depth"))};
  auto const null_frequency{state.get_float64("null_frequency")};

  data_profile table_profile;
  table_profile.set_distribution_params(cudf::type_id::LIST, distribution_id::UNIFORM, 0, 5);
  table_profile.set_list_depth(depth);
  table_profile.set_null_probability(null_frequency);
  return create_random_table({cudf::type_id::LIST}, table_size_bytes{size_bytes}, table_profile);
}

inline std::unique_ptr<cudf::table> create_structs_data(nvbench::state& state,
                                                        cudf::size_type const n_cols = 1)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{static_cast<cudf::size_type>(state.get_int64("NumRows"))};
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
  // Nest the child columns in a struct, then nest that struct column inside another
  // struct column up to the desired depth
  for (int i = 0; i < depth; i++) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 100 * (i + 1));
    std::generate_n(
      std::back_inserter(struct_validity), n_rows, [&]() { return bool_distribution(generator); });
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols), struct_validity);
    child_cols = std::vector<std::unique_ptr<cudf::column>>{};
    child_cols.push_back(struct_col.release());
  }

  // Create table view
  return std::make_unique<cudf::table>(std::move(child_cols));
}
