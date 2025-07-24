/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "join_common.hpp"

#include <cudf/join/conditional_join.hpp>

auto const CONDITIONAL_JOIN_SIZE_RANGE = std::vector<nvbench::int64_t>{1000, 100'000};
auto const num_keys                    = 1;

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_conditional_inner_join(nvbench::state& state,
                                    nvbench::type_list<nvbench::enum_type<Nullable>,
                                                       nvbench::enum_type<NullEquality>,
                                                       nvbench::enum_type<DataType>>)
{
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left,
                 cudf::table_view const& right,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::conditional_inner_join(left, right, binary_pred);
  };

  BM_join<Nullable, join_t::CONDITIONAL, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_conditional_left_join(nvbench::state& state,
                                   nvbench::type_list<nvbench::enum_type<Nullable>,
                                                      nvbench::enum_type<NullEquality>,
                                                      nvbench::enum_type<DataType>>)
{
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left,
                 cudf::table_view const& right,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::conditional_left_join(left, right, binary_pred);
  };

  BM_join<Nullable, join_t::CONDITIONAL, NullEquality>(state, dtypes, join);
}

NVBENCH_BENCH_TYPES(nvbench_conditional_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("conditional_inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", CONDITIONAL_JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", CONDITIONAL_JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_conditional_left_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("conditional_left_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", CONDITIONAL_JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", CONDITIONAL_JOIN_SIZE_RANGE);
