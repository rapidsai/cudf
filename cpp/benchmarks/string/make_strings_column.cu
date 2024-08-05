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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/pair.h>
#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

#include <vector>

static void BM_strings_column_construction(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const has_nulls = static_cast<bool>(state.get_int64("has_nulls"));

  constexpr int min_row_width = 0;
  constexpr int max_row_width = 50;
  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_row_width, max_row_width)
      .null_probability(has_nulls ? std::optional<double>{0.1} : std::nullopt);
  auto const data_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  using string_index_pair = thrust::pair<char const*, cudf::size_type>;
  auto const stream       = cudf::get_default_stream();
  auto input              = rmm::device_uvector<string_index_pair>(data_table->num_rows(), stream);
  auto const d_data_ptr =
    cudf::column_device_view::create(data_table->get_column(0).view(), stream);
  thrust::tabulate(rmm::exec_policy(stream),
                   input.begin(),
                   input.end(),
                   [data_col = *d_data_ptr] __device__(auto const idx) {
                     auto const row = data_col.element<cudf::string_view>(idx);
                     return string_index_pair{row.data(), row.size_bytes()};
                   });

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto const output = cudf::make_strings_column(input, stream);
  });
}

static void BM_strings_column_batch_construction(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const has_nulls  = static_cast<bool>(state.get_int64("has_nulls"));
  auto const batch_size = static_cast<cudf::size_type>(state.get_int64("batch_size"));

  constexpr int min_row_width = 0;
  constexpr int max_row_width = 50;
  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_row_width, max_row_width)
      .null_probability(has_nulls ? std::optional<double>{0.1} : std::nullopt);
  auto const data_table = create_random_table(
    cycle_dtypes({cudf::type_id::STRING}, batch_size), row_count{num_rows}, table_profile);

  using string_index_pair = thrust::pair<char const*, cudf::size_type>;
  auto const stream       = cudf::get_default_stream();
  auto input              = std::vector<rmm::device_uvector<string_index_pair>>{};
  input.reserve(batch_size);
  for (auto i = 0; i < batch_size; ++i) {
    auto const d_data_ptr =
      cudf::column_device_view::create(data_table->get_column(i).view(), stream);
    auto batch_input = rmm::device_uvector<string_index_pair>(data_table->num_rows(), stream);
    thrust::tabulate(rmm::exec_policy(stream),
                     batch_input.begin(),
                     batch_input.end(),
                     [data_col = *d_data_ptr] __device__(auto const idx) {
                       auto const row = data_col.element<cudf::string_view>(idx);
                       return string_index_pair{row.data(), row.size_bytes()};
                     });
    input.push_back(std::move(batch_input));
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    for (auto const& batch : input) {
      [[maybe_unused]] auto const output = cudf::make_strings_column(batch, stream);
    }
  });
}

NVBENCH_BENCH(BM_strings_column_construction)
  .set_name("strings_column_construction")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000, 200'000'000})
  .add_int64_axis("has_nulls", {0, 1});

NVBENCH_BENCH(BM_strings_column_batch_construction)
  .set_name("strings_column_batch_construction")
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 20'000'000})
  .add_int64_axis("has_nulls", {0, 1})
  .add_int64_axis("batch_size", {5, 10, 15, 20});
