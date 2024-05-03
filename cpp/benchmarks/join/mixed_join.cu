/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

template <typename Key, bool Nullable>
void nvbench_mixed_inner_join(nvbench::state& state,
                              nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    return cudf::mixed_inner_join(left_equality_input,
                                  right_equality_input,
                                  left_conditional_input,
                                  right_conditional_input,
                                  binary_pred,
                                  compare_nulls);
  };

  BM_join<Key, Nullable, join_t::MIXED>(state, join);
}

template <typename Key, bool Nullable>
void nvbench_mixed_left_join(nvbench::state& state,
                             nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    return cudf::mixed_left_join(left_equality_input,
                                 right_equality_input,
                                 left_conditional_input,
                                 right_conditional_input,
                                 binary_pred,
                                 compare_nulls);
  };

  BM_join<Key, Nullable, join_t::MIXED>(state, join);
}

template <typename Key, bool Nullable>
void nvbench_mixed_full_join(nvbench::state& state,
                             nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    return cudf::mixed_full_join(left_equality_input,
                                 right_equality_input,
                                 left_conditional_input,
                                 right_conditional_input,
                                 binary_pred,
                                 compare_nulls);
  };

  BM_join<Key, Nullable, join_t::MIXED>(state, join);
}

template <typename Key, bool Nullable>
void nvbench_mixed_left_semi_join(nvbench::state& state,
                                  nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    return cudf::mixed_left_semi_join(left_equality_input,
                                      right_equality_input,
                                      left_conditional_input,
                                      right_conditional_input,
                                      binary_pred,
                                      compare_nulls);
  };

  BM_join<Key, Nullable, join_t::MIXED>(state, join);
}

template <typename Key, bool Nullable>
void nvbench_mixed_left_anti_join(nvbench::state& state,
                                  nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    return cudf::mixed_left_anti_join(left_equality_input,
                                      right_equality_input,
                                      left_conditional_input,
                                      right_conditional_input,
                                      binary_pred,
                                      compare_nulls);
  };

  BM_join<Key, Nullable, join_t::MIXED>(state, join);
}

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("mixed_inner_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("mixed_left_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("mixed_full_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("mixed_left_semi_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("mixed_left_anti_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);
