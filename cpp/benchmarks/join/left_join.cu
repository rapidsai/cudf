/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common.hpp"

#include <cudf/join/filtered_join.hpp>
#include <cudf/utilities/default_stream.hpp>

auto const num_keys = 1;

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_left_anti_join(nvbench::state& state,
                            nvbench::type_list<nvbench::enum_type<Nullable>,
                                               nvbench::enum_type<NullEquality>,
                                               nvbench::enum_type<DataType>>)
{
  auto const num_operations   = static_cast<cudf::size_type>(state.get_int64("num_operations"));
  auto const reuse_left_table = state.get_string("reuse_table") == "left"
                                  ? cudf::set_as_build_table::LEFT
                                  : cudf::set_as_build_table::RIGHT;
  if (reuse_left_table == cudf::set_as_build_table::LEFT) {
    state.skip("Not yet implemented");
    return;
  }
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [num_operations, reuse_left_table](cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 cudf::null_equality compare_nulls) {
    cudf::filtered_join obj(right, compare_nulls, reuse_left_table, cudf::get_default_stream());
    for (auto i = 0; i < num_operations - 1; i++) {
      [[maybe_unused]] auto result = obj.anti_join(left);
    }
    return obj.anti_join(left);
  };

  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_left_semi_join(nvbench::state& state,
                            nvbench::type_list<nvbench::enum_type<Nullable>,
                                               nvbench::enum_type<NullEquality>,
                                               nvbench::enum_type<DataType>>)
{
  auto const num_operations   = static_cast<cudf::size_type>(state.get_int64("num_operations"));
  auto const reuse_left_table = state.get_string("reuse_table") == "left"
                                  ? cudf::set_as_build_table::LEFT
                                  : cudf::set_as_build_table::RIGHT;
  if (reuse_left_table == cudf::set_as_build_table::LEFT) {
    state.skip("Not yet implemented");
    return;
  }
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [num_operations, reuse_left_table](cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 cudf::null_equality compare_nulls) {
    cudf::filtered_join obj(right, compare_nulls, reuse_left_table, cudf::get_default_stream());
    for (auto i = 0; i < num_operations - 1; i++) {
      [[maybe_unused]] auto result = obj.semi_join(left);
    }
    return obj.semi_join(left);
  };
  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

NVBENCH_BENCH_TYPES(nvbench_left_anti_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("left_anti_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("num_operations", {4})
  .add_string_axis("reuse_table", {"left", "right"});

NVBENCH_BENCH_TYPES(nvbench_left_semi_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("left_semi_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("num_operations", {4})
  .add_string_axis("reuse_table", {"left", "right"});
