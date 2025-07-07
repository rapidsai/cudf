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
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>

template <typename Key, bool Nullable, cudf::null_equality NullEquality, join_t Algorithm>
void nvbench_inner_join(nvbench::state& state,
                        nvbench::type_list<Key,
                                           nvbench::enum_type<Nullable>,
                                           nvbench::enum_type<NullEquality>,
                                           nvbench::enum_type<Algorithm>>)
{
  if (Nullable == false && NullEquality == cudf::null_equality::UNEQUAL) {
    state.skip(
      "Since the keys are not nullable, how null entries are to be compared by the join algorithm "
      "is "
      "immaterial. Therefore, we skip running the benchmark when null equality is set to "
      "UNEQUAL since the performance numbers will be the same as when null equality is set to "
      "EQUAL.");
    return;
  }
  auto const multiplicity = static_cast<cudf::size_type>(state.get_int64("multiplicity"));
  auto hash_join          = [](cudf::table_view const& left_input,
                      cudf::table_view const& right_input,
                      cudf::null_equality compare_nulls) {
    return cudf::inner_join(left_input, right_input, compare_nulls);
  };
  auto sort_merge_join = [](cudf::table_view const& left_input,
                            cudf::table_view const& right_input,
                            cudf::null_equality compare_nulls) {
    auto smj = cudf::sort_merge_join(right_input, cudf::sorted::NO, compare_nulls);
    return smj.inner_join(left_input, cudf::sorted::NO);
  };
  if constexpr (Algorithm == join_t::HASH) {
    BM_join<Key, Nullable, Algorithm, NullEquality>(state, hash_join, multiplicity);
  } else if constexpr (Algorithm == join_t::SORT_MERGE) {
    BM_join<Key, Nullable, Algorithm, NullEquality>(state, sort_merge_join, multiplicity);
  }
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

using NULL_EQUALITY =
  nvbench::enum_type_list<cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL>;

using ALGORITHM = nvbench::enum_type_list<join_t::HASH, join_t::SORT_MERGE>;

NVBENCH_BENCH_TYPES(
  nvbench_inner_join,
  NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE, NULL_EQUALITY, ALGORITHM))
  .set_name("high_multiplicity_inner_join")
  .set_type_axes_names({"Key", "Nullable", "NullEquality", "Algorithm"})
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
