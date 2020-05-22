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

#include <benchmark/benchmark.h>

#include <cudf/column/column.hpp>

#include <cudf/copying.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <tests/utilities/column_wrapper.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

template <typename T>
void BM_contiguous_split_common(benchmark::State& state,
                                std::vector<T>& src_cols,
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

  std::vector<std::unique_ptr<cudf::column>> columns(src_cols.size());
  std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](T& in) {
    auto ret = in.release();
    ret->null_count();
    return ret;
  });
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
  srand(31337);
  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  std::vector<cudf::test::fixed_width_column_wrapper<int>> src_cols(num_cols);
  for (int idx = 0; idx < num_cols; idx++) {
    auto rand_elements =
      cudf::test::make_counting_transform_iterator(0, [](int i) { return rand(); });
    if (include_validity) {
      src_cols[idx] = cudf::test::fixed_width_column_wrapper<int>(
        rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<int>(rand_elements, rand_elements + num_rows);
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

  const int64_t string_len = 8;
  std::vector<const char*> h_strings{
    "aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee", "ffffffff", "gggggggg", "hhhhhhhh"};

  int64_t col_len_bytes = total_desired_bytes / num_cols;
  int64_t num_rows      = col_len_bytes / string_len;

  // generate input table
  srand(31337);
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  std::vector<cudf::test::strings_column_wrapper> src_cols;
  std::vector<const char*> one_col(num_rows);
  for (int64_t idx = 0; idx < num_cols; idx++) {
    // fill in a random set of strings
    for (int64_t s_idx = 0; s_idx < num_rows; s_idx++) {
      one_col[s_idx] = h_strings[rand_range(h_strings.size())];
    }
    if (include_validity) {
      src_cols.push_back(
        cudf::test::strings_column_wrapper(one_col.begin(), one_col.end(), valids));
    } else {
      src_cols.push_back(cudf::test::strings_column_wrapper(one_col.begin(), one_col.end()));
    }
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
