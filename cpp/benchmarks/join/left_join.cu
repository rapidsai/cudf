/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common.hpp"

#include <cudf/join/filtered_join.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/utilities/default_stream.hpp>

auto const num_keys = 1;

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_left_anti_join(nvbench::state& state,
                            nvbench::type_list<nvbench::enum_type<Nullable>,
                                               nvbench::enum_type<NullEquality>,
                                               nvbench::enum_type<DataType>>)
{
  auto const num_operations = static_cast<cudf::size_type>(state.get_int64("num_operations"));
  auto const left_size      = state.get_int64("left_size");
  auto const right_size     = state.get_int64("right_size");
  auto const join_type      = state.get_string("join_type");
  if (join_type == "mark_join" && (left_size > right_size || left_size > 100'000)) {
    state.skip("mark_join: build (left) should be smaller than probe (right) and <= 100K");
    return;
  }
  if (join_type == "filtered_join" && right_size > left_size) {
    state.skip("filtered_join: build (right) should be smaller than probe (left)");
    return;
  }
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [num_operations, &join_type](cudf::table_view const& left,
                                           cudf::table_view const& right,
                                           cudf::null_equality compare_nulls) {
    if (join_type == "mark_join") {
      cudf::mark_join obj(left, compare_nulls, cudf::get_default_stream());
      for (auto i = 0; i < num_operations - 1; i++) {
        [[maybe_unused]] auto result = obj.anti_join(right);
      }
      return obj.anti_join(right);
    } else {
      cudf::filtered_join obj(
        right, compare_nulls, cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
      for (auto i = 0; i < num_operations - 1; i++) {
        [[maybe_unused]] auto result = obj.anti_join(left);
      }
      return obj.anti_join(left);
    }
  };

  auto const skip_large_right = (join_type == "filtered_join");
  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join, 1, 0.3, skip_large_right);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_left_semi_join(nvbench::state& state,
                            nvbench::type_list<nvbench::enum_type<Nullable>,
                                               nvbench::enum_type<NullEquality>,
                                               nvbench::enum_type<DataType>>)
{
  auto const num_operations = static_cast<cudf::size_type>(state.get_int64("num_operations"));
  auto const left_size      = state.get_int64("left_size");
  auto const right_size     = state.get_int64("right_size");
  auto const join_type      = state.get_string("join_type");
  if (join_type == "mark_join" && (left_size > right_size || left_size > 100'000)) {
    state.skip("mark_join: build (left) should be smaller than probe (right) and <= 100K");
    return;
  }
  if (join_type == "filtered_join" && right_size > left_size) {
    state.skip("filtered_join: build (right) should be smaller than probe (left)");
    return;
  }
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [num_operations, &join_type](cudf::table_view const& left,
                                           cudf::table_view const& right,
                                           cudf::null_equality compare_nulls) {
    if (join_type == "mark_join") {
      cudf::mark_join obj(left, compare_nulls, cudf::get_default_stream());
      for (auto i = 0; i < num_operations - 1; i++) {
        [[maybe_unused]] auto result = obj.semi_join(right);
      }
      return obj.semi_join(right);
    } else {
      cudf::filtered_join obj(
        right, compare_nulls, cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
      for (auto i = 0; i < num_operations - 1; i++) {
        [[maybe_unused]] auto result = obj.semi_join(left);
      }
      return obj.semi_join(left);
    }
  };
  auto const skip_large_right = (join_type == "filtered_join");
  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join, 1, 0.3, skip_large_right);
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
  .add_string_axis("join_type", {"mark_join", "filtered_join"});

NVBENCH_BENCH_TYPES(nvbench_left_semi_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("left_semi_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("num_operations", {4})
  .add_string_axis("join_type", {"mark_join", "filtered_join"});
