/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/filling.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/tuple.h>

struct url_string_generator {
  char* chars;
  double esc_seq_chance;
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> esc_seq_dist;
  url_string_generator(char* c, double esc_seq_chance, thrust::minstd_rand& engine)
    : chars(c), esc_seq_chance(esc_seq_chance), engine(engine), esc_seq_dist(0, 1)
  {
  }

  __device__ void operator()(thrust::tuple<int64_t, int64_t> str_begin_end)
  {
    auto begin = thrust::get<0>(str_begin_end);
    auto end   = thrust::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      if (esc_seq_dist(engine) < esc_seq_chance and i < end - 3) {
        chars[i]     = '%';
        chars[i + 1] = '2';
        chars[i + 2] = '0';
        i += 2;
      } else {
        chars[i] = 'a';
      }
    }
  }
};

auto generate_column(cudf::size_type num_rows, cudf::size_type chars_per_row, double esc_seq_chance)
{
  std::vector<std::string> strings{std::string(chars_per_row, 'a')};
  auto col_1a     = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto table_a    = cudf::repeat(cudf::table_view{{col_1a}}, num_rows);
  auto result_col = std::move(table_a->release()[0]);  // string column with num_rows  aaa...
  auto chars_data = static_cast<char*>(result_col->mutable_view().head());
  auto offset_col = result_col->child(cudf::strings_column_view::offsets_column_index).view();
  auto offset_itr = cudf::detail::offsetalator_factory::make_input_iterator(offset_col);

  auto engine = thrust::default_random_engine{};
  thrust::for_each_n(thrust::device,
                     thrust::make_zip_iterator(offset_itr, offset_itr + 1),
                     num_rows,
                     url_string_generator{chars_data, esc_seq_chance, engine});
  return result_col;
}

class UrlDecode : public cudf::benchmark {};

void BM_url_decode(benchmark::State& state, int esc_seq_pct)
{
  cudf::size_type const num_rows      = state.range(0);
  cudf::size_type const chars_per_row = state.range(1);

  auto column       = generate_column(num_rows, chars_per_row, esc_seq_pct / 100.0);
  auto strings_view = cudf::strings_column_view(column->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    auto result = cudf::strings::url_decode(strings_view);
  }

  state.SetBytesProcessed(state.iterations() * num_rows *
                          (chars_per_row + sizeof(cudf::size_type)));
}

#define URLD_BENCHMARK_DEFINE(esc_seq_pct)                      \
  BENCHMARK_DEFINE_F(UrlDecode, esc_seq_pct)                    \
  (::benchmark::State & st) { BM_url_decode(st, esc_seq_pct); } \
  BENCHMARK_REGISTER_F(UrlDecode, esc_seq_pct)                  \
    ->Args({100000000, 10})                                     \
    ->Args({10000000, 100})                                     \
    ->Args({1000000, 1000})                                     \
    ->Unit(benchmark::kMillisecond)                             \
    ->UseManualTime();

URLD_BENCHMARK_DEFINE(10)
URLD_BENCHMARK_DEFINE(50)
