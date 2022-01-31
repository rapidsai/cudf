/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/detail/tagged_iterator.h>
#include <thrust/random.h>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

template <typename Tag, typename Iterator>
inline auto make_tagged_iterator(Iterator iter)
{
  return thrust::detail::tagged_iterator<Iterator, Tag>(iter);
}

template <typename T>
void BM_contiguous_split_common(benchmark::State& state,
                                std::vector<T>& columns,
                                int64_t num_rows,
                                int64_t num_splits,
                                int64_t bytes_total)
{
  // generate splits
  cudf::size_type split_stride = num_rows / num_splits;
  std::vector<cudf::size_type> splits;
  for (int idx = 0; idx < num_rows; idx += split_stride) {
    splits.push_back(std::min(idx + split_stride, static_cast<cudf::size_type>(num_rows)));
  }

  for (auto&& c : columns) {
    // computing the null count is not a part of the benchmark's target code path, and we want the
    // property to be pre-computed so that we measure the performance of only the intended code path
    [[maybe_unused]] auto const nulls = c->null_count();
  }

  cudf::table src_table(std::move(columns));

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    auto result = cudf::contiguous_split(src_table, splits);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_total);
}

class ContiguousSplit : public cudf::benchmark {
};

void BM_contiguous_split(benchmark::State& state)
{
  int64_t total_desired_bytes = state.range(0);
  cudf::size_type num_cols    = state.range(1);
  cudf::size_type num_splits  = state.range(2);
  bool include_validity       = state.range(3) == 0 ? false : true;

  cudf::size_type el_size = 4;  // ints and floats
  int64_t num_rows        = total_desired_bytes / (num_cols * el_size);

  // generate input table
  auto valids = thrust::constant_iterator<bool>(true);
  std::vector<std::unique_ptr<cudf::column>> src_cols(num_cols);
  for (int idx = 0; idx < num_cols; idx++) {
    auto rand_elements = make_tagged_iterator<thrust::device_system_tag>(
      cudf::detail::make_counting_transform_iterator(0u, [idx] __device__(uint32_t i) {
        thrust::default_random_engine rng(31337 + idx);
        thrust::uniform_int_distribution<uint32_t> dist;
        rng.discard(i);
        return dist(rng);
      }));
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows, valids)
          .release();
    } else {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows)
          .release();
    }
  }

  size_t total_bytes = total_desired_bytes;
  if (include_validity) { total_bytes += num_rows / (sizeof(cudf::bitmask_type) * 8); }

  BM_contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes);
}

class ContiguousSplitStrings : public cudf::benchmark {
};

int rand_range(int r)
{
  return static_cast<int>((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) *
                          (float)(r - 1));
}

void BM_contiguous_split_strings(benchmark::State& state)
{
  int64_t total_desired_bytes = state.range(0);
  cudf::size_type num_cols    = state.range(1);
  cudf::size_type num_splits  = state.range(2);
  bool include_validity       = state.range(3) == 0 ? false : true;
  using string_pair           = thrust::pair<const char*, cudf::size_type>;

  const int64_t string_len = 8;
  cudf::test::strings_column_wrapper w_strings(
    {"aaaaaaaa", "bbbbbbb", "cccccc", "ddddd", "eeee", "fff", "gg", "h", ""},
    {1, 1, 1, 1, 1, 1, 1, 1, 0});
  cudf::column_view d_strings = w_strings;
  int64_t col_len_bytes       = total_desired_bytes / num_cols;
  int64_t num_rows            = col_len_bytes / string_len;

  // generate input table
  std::vector<std::unique_ptr<cudf::column>> src_cols(num_cols);
  for (int64_t idx = 0; idx < num_cols; idx++) {
    // fill in a random set of strings
    auto rand_elements = make_tagged_iterator<thrust::device_system_tag>(
      cudf::detail::make_counting_transform_iterator(
        0u, [idx, sz = d_strings.size() - !include_validity] __device__(uint32_t i) {
          thrust::default_random_engine rng(31337 + idx);
          thrust::uniform_int_distribution<uint32_t> dist{0, sz - 1u};
          rng.discard(i);
          return dist(rng);
        }));
    auto d_elements =
      cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows);
    auto d_table = cudf::gather(
      cudf::table_view({d_strings}), d_elements, cudf::out_of_bounds_policy::DONT_CHECK);
    if (!include_validity) d_table->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
    src_cols[idx] = std::move(d_table->release()[0]);
  }

  size_t total_bytes = total_desired_bytes + (num_rows * sizeof(cudf::size_type));
  if (include_validity) { total_bytes += num_rows / (sizeof(cudf::bitmask_type) * 8); }

  BM_contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes);
}

#define CSBM_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ContiguousSplit, name)(::benchmark::State & state)      \
  {                                                                          \
    BM_contiguous_split(state);                                              \
  }                                                                          \
  BENCHMARK_REGISTER_F(ContiguousSplit, name)                                \
    ->Args({size, num_columns, num_splits, validity})                        \
    ->Unit(benchmark::kMillisecond)                                          \
    ->UseManualTime()                                                        \
    ->Iterations(8)
CSBM_BENCHMARK_DEFINE(6Gb512ColsNoValidity, (int64_t)6 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_BENCHMARK_DEFINE(6Gb512ColsValidity, (int64_t)6 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_BENCHMARK_DEFINE(6Gb10ColsNoValidity, (int64_t)6 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_BENCHMARK_DEFINE(6Gb10ColsValidity, (int64_t)6 * 1024 * 1024 * 1024, 10, 256, 1);

CSBM_BENCHMARK_DEFINE(4Gb512ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_BENCHMARK_DEFINE(4Gb512ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_BENCHMARK_DEFINE(4Gb10ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_BENCHMARK_DEFINE(46b10ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 256, 1);

CSBM_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 1);

#define CSBM_STRINGS_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ContiguousSplitStrings, name)(::benchmark::State & state)       \
  {                                                                                  \
    BM_contiguous_split_strings(state);                                              \
  }                                                                                  \
  BENCHMARK_REGISTER_F(ContiguousSplitStrings, name)                                 \
    ->Args({size, num_columns, num_splits, validity})                                \
    ->Unit(benchmark::kMillisecond)                                                  \
    ->UseManualTime()                                                                \
    ->Iterations(8)

CSBM_STRINGS_BENCHMARK_DEFINE(4Gb512ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(4Gb512ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_STRINGS_BENCHMARK_DEFINE(4Gb10ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(4Gb10ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 256, 1);

CSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 1);
