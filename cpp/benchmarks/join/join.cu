/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
void nvbench_inner_join(nvbench::state& state,
                        nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    cudf::hash_join hj_obj(left_input, compare_nulls, stream);
    return hj_obj.inner_join(right_input, std::nullopt, stream);
  };

  BM_join<key_type, payload_type, Nullable>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_left_join(nvbench::state& state,
                       nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    cudf::hash_join hj_obj(left_input, compare_nulls, stream);
    return hj_obj.left_join(right_input, std::nullopt, stream);
  };

  BM_join<key_type, payload_type, Nullable>(state, join);
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_full_join(nvbench::state& state,
                       nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  skip_helper(state);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    cudf::hash_join hj_obj(left_input, compare_nulls, stream);
    return hj_obj.full_join(right_input, std::nullopt, stream);
  };

  BM_join<key_type, payload_type, Nullable>(state, join);
}

// inner join -----------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("inner_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("inner_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("inner_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_inner_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("inner_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// left join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("left_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("left_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("left_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_left_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("left_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

// full join ------------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("full_join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("full_join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("full_join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size",
                  {100'000, 400'000, 10'000'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_full_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("full_join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});
