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
class ConditionalJoin : public cudf::benchmark {};

// For compatibility with the shared logic for equality (hash) joins, all of
// the join lambdas defined by these macros accept a null_equality parameter
// but ignore it (don't forward it to the underlying join implementation)
// because conditional joins do not use this parameter.
#define CONDITIONAL_INNER_JOIN_BENCHMARK_DEFINE(name, Key, Nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(ConditionalJoin, name, Key)            \
  (::benchmark::State & st)                                          \
  {                                                                  \
    auto join = [](cudf::table_view const& left,                     \
                   cudf::table_view const& right,                    \
                   cudf::ast::operation binary_pred,                 \
                   cudf::null_equality compare_nulls) {              \
      return cudf::conditional_inner_join(left, right, binary_pred); \
    };                                                               \
    BM_join<Key, Nullable, join_t::CONDITIONAL>(st, join);           \
  }

CONDITIONAL_INNER_JOIN_BENCHMARK_DEFINE(conditional_inner_join_32bit, int32_t, false);
CONDITIONAL_INNER_JOIN_BENCHMARK_DEFINE(conditional_inner_join_64bit, int64_t, false);
CONDITIONAL_INNER_JOIN_BENCHMARK_DEFINE(conditional_inner_join_32bit_nulls, int32_t, true);
CONDITIONAL_INNER_JOIN_BENCHMARK_DEFINE(conditional_inner_join_64bit_nulls, int64_t, true);

#define CONDITIONAL_LEFT_JOIN_BENCHMARK_DEFINE(name, Key, Nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(ConditionalJoin, name, Key)           \
  (::benchmark::State & st)                                         \
  {                                                                 \
    auto join = [](cudf::table_view const& left,                    \
                   cudf::table_view const& right,                   \
                   cudf::ast::operation binary_pred,                \
                   cudf::null_equality compare_nulls) {             \
      return cudf::conditional_left_join(left, right, binary_pred); \
    };                                                              \
    BM_join<Key, Nullable, join_t::CONDITIONAL>(st, join);          \
  }

CONDITIONAL_LEFT_JOIN_BENCHMARK_DEFINE(conditional_left_join_32bit, int32_t, false);
CONDITIONAL_LEFT_JOIN_BENCHMARK_DEFINE(conditional_left_join_64bit, int64_t, false);
CONDITIONAL_LEFT_JOIN_BENCHMARK_DEFINE(conditional_left_join_32bit_nulls, int32_t, true);
CONDITIONAL_LEFT_JOIN_BENCHMARK_DEFINE(conditional_left_join_64bit_nulls, int64_t, true);

#define CONDITIONAL_FULL_JOIN_BENCHMARK_DEFINE(name, Key, Nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(ConditionalJoin, name, Key)           \
  (::benchmark::State & st)                                         \
  {                                                                 \
    auto join = [](cudf::table_view const& left,                    \
                   cudf::table_view const& right,                   \
                   cudf::ast::operation binary_pred,                \
                   cudf::null_equality compare_nulls) {             \
      return cudf::conditional_full_join(left, right, binary_pred); \
    };                                                              \
    BM_join<Key, Nullable, join_t::CONDITIONAL>(st, join);          \
  }

CONDITIONAL_FULL_JOIN_BENCHMARK_DEFINE(conditional_full_join_32bit, int32_t, false);
CONDITIONAL_FULL_JOIN_BENCHMARK_DEFINE(conditional_full_join_64bit, int64_t, false);
CONDITIONAL_FULL_JOIN_BENCHMARK_DEFINE(conditional_full_join_32bit_nulls, int32_t, true);
CONDITIONAL_FULL_JOIN_BENCHMARK_DEFINE(conditional_full_join_64bit_nulls, int64_t, true);

#define CONDITIONAL_LEFT_ANTI_JOIN_BENCHMARK_DEFINE(name, Key, Nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(ConditionalJoin, name, Key)                \
  (::benchmark::State & st)                                              \
  {                                                                      \
    auto join = [](cudf::table_view const& left,                         \
                   cudf::table_view const& right,                        \
                   cudf::ast::operation binary_pred,                     \
                   cudf::null_equality compare_nulls) {                  \
      return cudf::conditional_left_anti_join(left, right, binary_pred); \
    };                                                                   \
    BM_join<Key, Nullable, join_t::CONDITIONAL>(st, join);               \
  }

CONDITIONAL_LEFT_ANTI_JOIN_BENCHMARK_DEFINE(conditional_left_anti_join_32bit, int32_t, false);
CONDITIONAL_LEFT_ANTI_JOIN_BENCHMARK_DEFINE(conditional_left_anti_join_64bit, int64_t, false);
CONDITIONAL_LEFT_ANTI_JOIN_BENCHMARK_DEFINE(conditional_left_anti_join_32bit_nulls, int32_t, true);
CONDITIONAL_LEFT_ANTI_JOIN_BENCHMARK_DEFINE(conditional_left_anti_join_64bit_nulls, int64_t, true);

#define CONDITIONAL_LEFT_SEMI_JOIN_BENCHMARK_DEFINE(name, Key, Nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(ConditionalJoin, name, Key)                \
  (::benchmark::State & st)                                              \
  {                                                                      \
    auto join = [](cudf::table_view const& left,                         \
                   cudf::table_view const& right,                        \
                   cudf::ast::operation binary_pred,                     \
                   cudf::null_equality compare_nulls) {                  \
      return cudf::conditional_left_semi_join(left, right, binary_pred); \
    };                                                                   \
    BM_join<Key, Nullable, join_t::CONDITIONAL>(st, join);               \
  }

CONDITIONAL_LEFT_SEMI_JOIN_BENCHMARK_DEFINE(conditional_left_semi_join_32bit, int32_t, false);
CONDITIONAL_LEFT_SEMI_JOIN_BENCHMARK_DEFINE(conditional_left_semi_join_64bit, int64_t, false);
CONDITIONAL_LEFT_SEMI_JOIN_BENCHMARK_DEFINE(conditional_left_semi_join_32bit_nulls, int32_t, true);
CONDITIONAL_LEFT_SEMI_JOIN_BENCHMARK_DEFINE(conditional_left_semi_join_64bit_nulls, int64_t, true);

// inner join -----------------------------------------------------------------------
BENCHMARK_REGISTER_F(ConditionalJoin, conditional_inner_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({400'000, 100'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_inner_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({400'000, 100'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_inner_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({400'000, 100'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_inner_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({400'000, 100'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

// left join -----------------------------------------------------------------------
BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

// full join -----------------------------------------------------------------------
BENCHMARK_REGISTER_F(ConditionalJoin, conditional_full_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_full_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_full_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_full_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

// left anti-join -------------------------------------------------------------
BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_anti_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_anti_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_anti_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_anti_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

// left semi-join -------------------------------------------------------------
BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_semi_join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_semi_join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_semi_join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(ConditionalJoin, conditional_left_semi_join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->UseManualTime();
