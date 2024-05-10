/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

template <typename Key>
class Join : public cudf::benchmark {};

#define LEFT_ANTI_JOIN_BENCHMARK_DEFINE(name, Key, Nullable)   \
  BENCHMARK_TEMPLATE_DEFINE_F(Join, name, Key)                 \
  (::benchmark::State & st)                                    \
  {                                                            \
    auto join = [](cudf::table_view const& left,               \
                   cudf::table_view const& right,              \
                   cudf::null_equality compare_nulls) {        \
      return cudf::left_anti_join(left, right, compare_nulls); \
    };                                                         \
    BM_join<Key, Nullable>(st, join);                          \
  }

LEFT_ANTI_JOIN_BENCHMARK_DEFINE(left_anti_join_32bit, int32_t, false);
LEFT_ANTI_JOIN_BENCHMARK_DEFINE(left_anti_join_64bit, int64_t, false);
LEFT_ANTI_JOIN_BENCHMARK_DEFINE(left_anti_join_32bit_nulls, int32_t, true);
LEFT_ANTI_JOIN_BENCHMARK_DEFINE(left_anti_join_64bit_nulls, int64_t, true);

#define LEFT_SEMI_JOIN_BENCHMARK_DEFINE(name, Key, Nullable)   \
  BENCHMARK_TEMPLATE_DEFINE_F(Join, name, Key)                 \
  (::benchmark::State & st)                                    \
  {                                                            \
    auto join = [](cudf::table_view const& left,               \
                   cudf::table_view const& right,              \
                   cudf::null_equality compare_nulls) {        \
      return cudf::left_semi_join(left, right, compare_nulls); \
    };                                                         \
    BM_join<Key, Nullable>(st, join);                          \
  }

LEFT_SEMI_JOIN_BENCHMARK_DEFINE(left_semi_join_32bit, int32_t, false);
LEFT_SEMI_JOIN_BENCHMARK_DEFINE(left_semi_join_64bit, int64_t, false);
LEFT_SEMI_JOIN_BENCHMARK_DEFINE(left_semi_join_32bit_nulls, int32_t, true);
LEFT_SEMI_JOIN_BENCHMARK_DEFINE(left_semi_join_64bit_nulls, int64_t, true);

// left anti-join -------------------------------------------------------------
BENCHMARK_REGISTER_F(Join, left_anti_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_anti_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_anti_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_anti_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();

// left semi-join -------------------------------------------------------------
BENCHMARK_REGISTER_F(Join, left_semi_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_semi_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_semi_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, left_semi_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();
