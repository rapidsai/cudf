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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_word_minhash(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const seed_count = static_cast<cudf::size_type>(state.get_int64("seed_count"));
  auto const base64     = state.get_int64("hash_type") == 64;

  data_profile const strings_profile =
    data_profile_builder().distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, 5);
  auto strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);

  auto const num_offsets = (num_rows / row_width) + 1;
  auto offsets           = cudf::sequence(
    num_offsets, cudf::numeric_scalar<int32_t>(0), cudf::numeric_scalar<int32_t>(row_width));

  auto source = cudf::make_lists_column(num_offsets - 1,
                                        std::move(offsets),
                                        std::move(strings_table->release().front()),
                                        0,
                                        rmm::device_buffer{});

  data_profile const seeds_profile = data_profile_builder().null_probability(0).distribution(
    cudf::type_to_id<cudf::hash_value_type>(), distribution_id::NORMAL, 0, 4);
  auto const seed_type   = base64 ? cudf::type_id::UINT64 : cudf::type_id::UINT32;
  auto const seeds_table = create_random_table({seed_type}, row_count{seed_count}, seeds_profile);
  auto seeds             = seeds_table->get_column(0);
  seeds.set_null_mask(rmm::device_buffer{}, 0);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  cudf::strings_column_view input(cudf::lists_column_view(source->view()).child());
  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);  // output are hashes

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = base64 ? nvtext::word_minhash64(source->view(), seeds.view())
                         : nvtext::word_minhash(source->view(), seeds.view());
  });
}

NVBENCH_BENCH(bench_word_minhash)
  .set_name("word_minhash")
  .add_int64_axis("num_rows", {131072, 262144, 524288, 1048576, 2097152})
  .add_int64_axis("row_width", {10, 100, 1000})
  .add_int64_axis("seed_count", {2, 25})
  .add_int64_axis("hash_type", {32, 64});
