/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/search.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

namespace {
template <typename Type>
std::unique_ptr<cudf::column> create_column_data(cudf::size_type n_rows, bool has_nulls = false)
{
  data_profile profile = data_profile_builder().cardinality(0).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 1000);
  profile.set_null_probability(has_nulls ? std::optional{0.1} : std::nullopt);

  return create_random_column(cudf::type_to_id<Type>(), row_count{n_rows}, profile);
}

}  // namespace

static void nvbench_contains_scalar(nvbench::state& state)
{
  using Type = int;

  auto const has_nulls = static_cast<bool>(state.get_int64("has_nulls"));
  auto const size      = state.get_int64("data_size");

  auto const haystack = create_column_data<Type>(size, has_nulls);
  auto const needle   = cudf::make_fixed_width_scalar<Type>(size / 2);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const stream_view             = rmm::cuda_stream_view{launch.get_stream()};
    [[maybe_unused]] auto const result = cudf::detail::contains(*haystack, *needle, stream_view);
  });
}

NVBENCH_BENCH(nvbench_contains_scalar)
  .set_name("contains_scalar")
  .add_int64_power_of_two_axis("data_size", {10, 12, 14, 16, 18, 20, 22, 24, 26})
  .add_int64_axis("has_nulls", {0, 1});
