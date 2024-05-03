/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

template <typename Key, bool Nullable>
void distinct_inner_join(nvbench::state& state,
                         nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& build_input,
                 cudf::table_view const& probe_input,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    auto const has_nulls =
      cudf::has_nested_nulls(build_input) || cudf::has_nested_nulls(probe_input)
        ? cudf::nullable_join::YES
        : cudf::nullable_join::NO;
    auto hj_obj = cudf::distinct_hash_join<cudf::has_nested::NO>{
      build_input, probe_input, has_nulls, compare_nulls, stream};
    return hj_obj.inner_join(stream);
  };

  BM_join<Key, Nullable>(state, join);
}

template <typename Key, bool Nullable>
void distinct_left_join(nvbench::state& state,
                        nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& build_input,
                 cudf::table_view const& probe_input,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    auto const has_nulls =
      cudf::has_nested_nulls(build_input) || cudf::has_nested_nulls(probe_input)
        ? cudf::nullable_join::YES
        : cudf::nullable_join::NO;
    auto hj_obj = cudf::distinct_hash_join<cudf::has_nested::NO>{
      build_input, probe_input, has_nulls, compare_nulls, stream};
    return hj_obj.left_join(stream);
  };

  BM_join<Key, Nullable>(state, join);
}

NVBENCH_BENCH_TYPES(distinct_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("distinct_inner_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(distinct_left_join, NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("distinct_left_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);
