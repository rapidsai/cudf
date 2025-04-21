/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmarks/join/join_common.hpp>

#include <cudf/join/join.hpp>

void nvbench_inner_join(nvbench::state& state)
{
  auto const multiplicity = static_cast<cudf::size_type>(state.get_int64("multiplicity"));
  auto join               = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join<nvbench::int64_t, false>(state, join, multiplicity);
}

void nvbench_left_join(nvbench::state& state)
{
  auto const multiplicity = static_cast<cudf::size_type>(state.get_int64("multiplicity"));
  auto join               = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::left_join(left_input, right_input, compare_nulls);
  };
  BM_join<nvbench::int64_t, false>(state, join, multiplicity);
}

void nvbench_full_join(nvbench::state& state)
{
  auto const multiplicity = static_cast<cudf::size_type>(state.get_int64("multiplicity"));
  auto join               = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::full_join(left_input, right_input, compare_nulls);
  };
  BM_join<nvbench::int64_t, false>(state, join, multiplicity);
}

NVBENCH_BENCH(nvbench_inner_join)
  .set_name("high_multiplicity_inner_join")
  .add_int64_axis("left_size", {100'000})
  .add_int64_axis("right_size", {100'000})
  .add_int64_axis("multiplicity", {10, 20, 50, 100, 1'000, 10'000, 50'000});

NVBENCH_BENCH(nvbench_left_join)
  .set_name("high_multiplicity_left_join")
  .add_int64_axis("left_size", {100'000})
  .add_int64_axis("right_size", {100'000})
  .add_int64_axis("multiplicity", {10, 20, 50, 100, 1'000, 10'000, 50'000});

NVBENCH_BENCH(nvbench_full_join)
  .set_name("high_multiplicity_full_join")
  .add_int64_axis("left_size", {100'000})
  .add_int64_axis("right_size", {100'000})
  .add_int64_axis("multiplicity", {10, 20, 50, 100, 1'000, 10'000, 50'000});
