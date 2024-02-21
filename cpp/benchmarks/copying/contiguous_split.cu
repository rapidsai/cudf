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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>

#include <thrust/iterator/counting_iterator.h>

void contiguous_split(cudf::table_view const& src_table, std::vector<cudf::size_type> const& splits)
{
  auto result = cudf::contiguous_split(src_table, splits);
}

void chunked_pack(cudf::table_view const& src_table, std::vector<cudf::size_type> const&)
{
  auto const mr     = rmm::mr::get_current_device_resource();
  auto const stream = cudf::get_default_stream();
  auto user_buffer  = rmm::device_uvector<std::uint8_t>(100L * 1024 * 1024, stream, mr);
  auto chunked_pack = cudf::chunked_pack::create(src_table, user_buffer.size(), mr);
  while (chunked_pack->has_next()) {
    auto iter_size = chunked_pack->next(user_buffer);
  }
  stream.synchronize();
}

template <typename T, typename ContigSplitImpl>
void BM_contiguous_split_common(benchmark::State& state,
                                std::vector<T>& src_cols,
                                int64_t num_rows,
                                int64_t num_splits,
                                int64_t bytes_total,
                                ContigSplitImpl& impl)
{
  // generate splits
  std::vector<cudf::size_type> splits;
  if (num_splits > 0) {
    cudf::size_type const split_stride = num_rows / num_splits;
    // start after the first element.
    auto iter = thrust::make_counting_iterator(1);
    splits.reserve(num_splits);
    std::transform(iter,
                   iter + num_splits,
                   std::back_inserter(splits),
                   [split_stride, num_rows](cudf::size_type i) {
                     return std::min(i * split_stride, static_cast<cudf::size_type>(num_rows));
                   });
  }

  for (auto const& col : src_cols)
    // computing the null count is not a part of the benchmark's target code path, and we want the
    // property to be pre-computed so that we measure the performance of only the intended code path
    [[maybe_unused]]
    auto const nulls = col->null_count();

  auto const src_table = cudf::table(std::move(src_cols));

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    impl(src_table, splits);
  }

  // it's 2x bytes_total because we're both reading and writing.
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes_total * 2);
}

class ContiguousSplit : public cudf::benchmark {};
class ChunkedPack : public cudf::benchmark {};

template <typename ContiguousSplitImpl>
void BM_contiguous_split(benchmark::State& state, ContiguousSplitImpl& impl)
{
  int64_t const total_desired_bytes = state.range(0);
  cudf::size_type const num_cols    = state.range(1);
  cudf::size_type const num_splits  = state.range(2);
  bool const include_validity       = state.range(3) != 0;

  cudf::size_type el_size = 4;  // ints and floats
  int64_t const num_rows  = total_desired_bytes / (num_cols * el_size);

  // generate input table
  auto builder = data_profile_builder().cardinality(0).distribution<int>(cudf::type_id::INT32,
                                                                         distribution_id::UNIFORM);
  if (not include_validity) builder.no_validity();

  auto src_cols = create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                                      row_count{static_cast<cudf::size_type>(num_rows)},
                                      data_profile{builder})
                    ->release();

  int64_t const total_bytes =
    total_desired_bytes +
    (include_validity ? (max(int64_t{1}, (num_rows / 32)) * sizeof(cudf::bitmask_type) * num_cols)
                      : 0);

  BM_contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, impl);
}

class ContiguousSplitStrings : public cudf::benchmark {};
class ChunkedPackStrings : public cudf::benchmark {};

template <typename ContiguousSplitImpl>
void BM_contiguous_split_strings(benchmark::State& state, ContiguousSplitImpl& impl)
{
  int64_t const total_desired_bytes = state.range(0);
  cudf::size_type const num_cols    = state.range(1);
  cudf::size_type const num_splits  = state.range(2);
  bool const include_validity       = state.range(3) != 0;

  constexpr int64_t string_len = 8;
  std::vector<char const*> h_strings{
    "aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee", "ffffffff", "gggggggg", "hhhhhhhh"};

  int64_t const col_len_bytes = total_desired_bytes / num_cols;
  int64_t const num_rows      = col_len_bytes / string_len;

  // generate input table
  data_profile profile = data_profile_builder().no_validity().cardinality(0).distribution(
    cudf::type_id::INT32,
    distribution_id::UNIFORM,
    0ul,
    include_validity ? h_strings.size() * 2 : h_strings.size() - 1);  // out of bounds nullified
  cudf::test::strings_column_wrapper one_col(h_strings.begin(), h_strings.end());
  std::vector<std::unique_ptr<cudf::column>> src_cols(num_cols);
  for (int64_t idx = 0; idx < num_cols; idx++) {
    auto random_indices = create_random_column(
      cudf::type_id::INT32, row_count{static_cast<cudf::size_type>(num_rows)}, profile);
    auto str_table = cudf::gather(cudf::table_view{{one_col}},
                                  *random_indices,
                                  (include_validity ? cudf::out_of_bounds_policy::NULLIFY
                                                    : cudf::out_of_bounds_policy::DONT_CHECK));
    src_cols[idx]  = std::move(str_table->release()[0]);
  }

  int64_t const total_bytes =
    total_desired_bytes + ((num_rows + 1) * sizeof(cudf::size_type)) +
    (include_validity ? (max(int64_t{1}, (num_rows / 32)) * sizeof(cudf::bitmask_type) * num_cols)
                      : 0);

  BM_contiguous_split_common(state, src_cols, num_rows, num_splits, total_bytes, impl);
}

#define CSBM_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ContiguousSplit, name)(::benchmark::State & state)      \
  {                                                                          \
    BM_contiguous_split(state, contiguous_split);                            \
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
CSBM_BENCHMARK_DEFINE(4Gb10ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 256, 1);
CSBM_BENCHMARK_DEFINE(4Gb4ColsNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 1);
CSBM_BENCHMARK_DEFINE(4Gb4ColsValidityNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 1);

CSBM_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 1);
CSBM_BENCHMARK_DEFINE(1Gb1ColNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 1);
CSBM_BENCHMARK_DEFINE(1Gb1ColValidityNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 1);

#define CSBM_STRINGS_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ContiguousSplitStrings, name)(::benchmark::State & state)       \
  {                                                                                  \
    BM_contiguous_split_strings(state, contiguous_split);                            \
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
CSBM_STRINGS_BENCHMARK_DEFINE(4Gb4ColsNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(4Gb4ColsValidityNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 1);

CSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 256, 1);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 256, 1);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb1ColNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 0);
CSBM_STRINGS_BENCHMARK_DEFINE(1Gb1ColValidityNoSplits, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 1);

#define CCSBM_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ChunkedPack, name)(::benchmark::State & state)           \
  {                                                                           \
    BM_contiguous_split(state, chunked_pack);                                 \
  }                                                                           \
  BENCHMARK_REGISTER_F(ChunkedPack, name)                                     \
    ->Args({size, num_columns, num_splits, validity})                         \
    ->Unit(benchmark::kMillisecond)                                           \
    ->UseManualTime()                                                         \
    ->Iterations(8)
CCSBM_BENCHMARK_DEFINE(6Gb512ColsNoValidity, (int64_t)6 * 1024 * 1024 * 1024, 512, 0, 0);
CCSBM_BENCHMARK_DEFINE(6Gb512ColsValidity, (int64_t)6 * 1024 * 1024 * 1024, 512, 0, 1);
CCSBM_BENCHMARK_DEFINE(6Gb10ColsNoValidity, (int64_t)6 * 1024 * 1024 * 1024, 10, 0, 0);
CCSBM_BENCHMARK_DEFINE(6Gb10ColsValidity, (int64_t)6 * 1024 * 1024 * 1024, 10, 0, 1);

CCSBM_BENCHMARK_DEFINE(4Gb512ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 0, 0);
CCSBM_BENCHMARK_DEFINE(4Gb512ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 0, 1);
CCSBM_BENCHMARK_DEFINE(4Gb10ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 0, 0);
CCSBM_BENCHMARK_DEFINE(4Gb10ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 0, 1);
CCSBM_BENCHMARK_DEFINE(4Gb4ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 1);

CCSBM_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 0, 0);
CCSBM_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 0, 1);
CCSBM_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 0, 0);
CCSBM_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 0, 1);
CCSBM_BENCHMARK_DEFINE(1Gb1ColValidity, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 1);

#define CCSBM_STRINGS_BENCHMARK_DEFINE(name, size, num_columns, num_splits, validity) \
  BENCHMARK_DEFINE_F(ChunkedPackStrings, name)(::benchmark::State & state)            \
  {                                                                                   \
    BM_contiguous_split_strings(state, chunked_pack);                                 \
  }                                                                                   \
  BENCHMARK_REGISTER_F(ChunkedPackStrings, name)                                      \
    ->Args({size, num_columns, num_splits, validity})                                 \
    ->Unit(benchmark::kMillisecond)                                                   \
    ->UseManualTime()                                                                 \
    ->Iterations(8)

CCSBM_STRINGS_BENCHMARK_DEFINE(4Gb512ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 0, 0);
CCSBM_STRINGS_BENCHMARK_DEFINE(4Gb512ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 512, 0, 1);
CCSBM_STRINGS_BENCHMARK_DEFINE(4Gb10ColsNoValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 0, 0);
CCSBM_STRINGS_BENCHMARK_DEFINE(4Gb10ColsValidity, (int64_t)4 * 1024 * 1024 * 1024, 10, 0, 1);
CCSBM_STRINGS_BENCHMARK_DEFINE(4Gb4ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 4, 0, 1);

CCSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 0, 0);
CCSBM_STRINGS_BENCHMARK_DEFINE(1Gb512ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 512, 0, 1);
CCSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsNoValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 0, 0);
CCSBM_STRINGS_BENCHMARK_DEFINE(1Gb10ColsValidity, (int64_t)1 * 1024 * 1024 * 1024, 10, 0, 1);
CCSBM_STRINGS_BENCHMARK_DEFINE(1Gb1ColValidity, (int64_t)1 * 1024 * 1024 * 1024, 1, 0, 1);
