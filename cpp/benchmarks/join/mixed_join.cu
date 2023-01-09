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

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/join/join_common.hpp>

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_mixed_inner_join(
  nvbench::state& state, nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

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

  BM_join<key_type, payload_type, Nullable, join_t::MIXED>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_mixed_left_join(
  nvbench::state& state, nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

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

  BM_join<key_type, payload_type, Nullable, join_t::MIXED>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_mixed_full_join(
  nvbench::state& state, nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

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

  BM_join<key_type, payload_type, Nullable, join_t::MIXED>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_mixed_left_semi_join(
  nvbench::state& state, nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

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

  BM_join<key_type, payload_type, Nullable, join_t::MIXED>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_mixed_left_anti_join(
  nvbench::state& state, nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

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

  BM_join<key_type, payload_type, Nullable, join_t::MIXED>(state, join);
}

// inner join -----------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_inner_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_inner_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("inner_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_inner_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// left join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// full join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_full_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_full_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_full_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_full_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// left semi join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_semi_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_semi_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_semi_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_semi_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// left anti join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_anti_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("mixed_left_anti_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_anti_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("mixed_left_anti_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});
