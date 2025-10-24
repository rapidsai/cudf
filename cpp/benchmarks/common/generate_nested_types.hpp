/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "generate_input.hpp"

#include <cudf_test/column_wrapper.hpp>

// This error appears in GCC 11.3 and may be a compiler bug or nvbench bug.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <nvbench/nvbench.cuh>
#pragma GCC diagnostic pop

#include <random>

inline std::unique_ptr<cudf::table> create_lists_data(nvbench::state& state,
                                                      cudf::size_type const num_columns = 1,
                                                      cudf::size_type const min_val     = 0,
                                                      cudf::size_type const max_val     = 5)
{
  size_t const size_bytes(state.get_int64("size_bytes"));
  cudf::size_type const depth{static_cast<cudf::size_type>(state.get_int64("depth"))};
  auto const null_frequency{state.get_float64("null_frequency")};

  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::LIST, distribution_id::UNIFORM, min_val, max_val);
  table_profile.set_list_depth(depth);
  table_profile.set_null_probability(null_frequency);
  return create_random_table(std::vector<cudf::type_id>(num_columns, cudf::type_id::LIST),
                             table_size_bytes{size_bytes},
                             table_profile);
}

inline std::unique_ptr<cudf::table> create_structs_data(nvbench::state& state,
                                                        cudf::size_type const n_cols = 1)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.get_int64("NumRows"))};
  cudf::size_type const depth{static_cast<cudf::size_type>(state.get_int64("Depth"))};
  bool const nulls{static_cast<bool>(state.get_int64("Nulls"))};

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
