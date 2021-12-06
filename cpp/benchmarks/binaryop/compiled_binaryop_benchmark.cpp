/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>

#include <thrust/iterator/counting_iterator.h>

template <typename TypeLhs, typename TypeRhs, typename TypeOut, cudf::binary_operator>
class COMPILED_BINARYOP : public cudf::benchmark {
};

template <typename TypeLhs, typename TypeRhs, typename TypeOut>
void BM_compiled_binaryop(benchmark::State& state, cudf::binary_operator binop)
{
  const cudf::size_type column_size{(cudf::size_type)state.range(0)};

  auto data_it = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<TypeLhs> input1(data_it, data_it + column_size);
  cudf::test::fixed_width_column_wrapper<TypeRhs> input2(data_it, data_it + column_size);

  auto lhs          = cudf::column_view(input1);
  auto rhs          = cudf::column_view(input2);
  auto output_dtype = cudf::data_type(cudf::type_to_id<TypeOut>());

  // Call once for hot cache.
  cudf::binary_operation(lhs, rhs, binop, output_dtype);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    cudf::binary_operation(lhs, rhs, binop, output_dtype);
  }
}

// TODO tparam boolean for null.
#define BINARYOP_BENCHMARK_DEFINE(name, TypeLhs, TypeRhs, binop, TypeOut)              \
  BENCHMARK_TEMPLATE_DEFINE_F(                                                         \
    COMPILED_BINARYOP, name, TypeLhs, TypeRhs, TypeOut, cudf::binary_operator::binop)  \
  (::benchmark::State & st)                                                            \
  {                                                                                    \
    BM_compiled_binaryop<TypeLhs, TypeRhs, TypeOut>(st, cudf::binary_operator::binop); \
  }                                                                                    \
  BENCHMARK_REGISTER_F(COMPILED_BINARYOP, name)                                        \
    ->Unit(benchmark::kMicrosecond)                                                    \
    ->UseManualTime()                                                                  \
    ->Arg(10000)      /* 10k */                                                        \
    ->Arg(100000)     /* 100k */                                                       \
    ->Arg(1000000)    /* 1M */                                                         \
    ->Arg(10000000)   /* 10M */                                                        \
    ->Arg(100000000); /* 100M */

using namespace cudf;
using namespace numeric;

// clang-format off
BINARYOP_BENCHMARK_DEFINE(ADD_1,          float,        float,        ADD,                  float);
BINARYOP_BENCHMARK_DEFINE(ADD_2,          timestamp_s,  duration_s,   ADD,                  timestamp_s);
BINARYOP_BENCHMARK_DEFINE(SUB_1,          duration_s,   duration_D,   SUB,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(SUB_2,          int64_t,      int64_t,      SUB,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(MUL_1,          float,        float,        MUL,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(MUL_2,          duration_s,   int64_t,      MUL,                  duration_s);
BINARYOP_BENCHMARK_DEFINE(DIV_1,          int64_t,      int64_t,      DIV,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(DIV_2,          duration_ms,  int32_t,      DIV,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(TRUE_DIV,       int64_t,      int64_t,      TRUE_DIV,             int64_t);
BINARYOP_BENCHMARK_DEFINE(FLOOR_DIV,      int64_t,      int64_t,      FLOOR_DIV,            int64_t);
BINARYOP_BENCHMARK_DEFINE(MOD_1,          double,       double,       MOD,                  double);
BINARYOP_BENCHMARK_DEFINE(MOD_2,          duration_ms,  int64_t,      MOD,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(PMOD,           int32_t,      int64_t,      PMOD,                 double);
BINARYOP_BENCHMARK_DEFINE(PYMOD,          int32_t,      uint8_t,      PYMOD,                int64_t);
BINARYOP_BENCHMARK_DEFINE(POW,            int64_t,      int64_t,      POW,                  double);
BINARYOP_BENCHMARK_DEFINE(LOG_BASE,       float,        double,       LOG_BASE,             double);
BINARYOP_BENCHMARK_DEFINE(ATAN2,          float,        double,       ATAN2,                double);
BINARYOP_BENCHMARK_DEFINE(SHIFT_LEFT,     int,          int,          SHIFT_LEFT,           int);
BINARYOP_BENCHMARK_DEFINE(SHIFT_RIGHT,    int16_t,      int64_t,      SHIFT_RIGHT,          int);
BINARYOP_BENCHMARK_DEFINE(USHIFT_RIGHT,   int64_t,      int32_t,      SHIFT_RIGHT_UNSIGNED, int64_t);
BINARYOP_BENCHMARK_DEFINE(BITWISE_AND,    int64_t,      int32_t,      BITWISE_AND,          int16_t);
BINARYOP_BENCHMARK_DEFINE(BITWISE_OR,     int16_t,      int32_t,      BITWISE_OR,           int64_t);
BINARYOP_BENCHMARK_DEFINE(BITWISE_XOR,    int16_t,      int64_t,      BITWISE_XOR,          int32_t);
BINARYOP_BENCHMARK_DEFINE(LOGICAL_AND,    double,       int8_t,       LOGICAL_AND,          bool);
BINARYOP_BENCHMARK_DEFINE(LOGICAL_OR,     int16_t,      int64_t,      LOGICAL_OR,           bool);
BINARYOP_BENCHMARK_DEFINE(EQUAL_1,        int32_t,      int64_t,      EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(EQUAL_2,        duration_ms,  duration_ns,  EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(NOT_EQUAL,      decimal32,    decimal32,    NOT_EQUAL,            bool);
BINARYOP_BENCHMARK_DEFINE(LESS,           timestamp_s,  timestamp_s,  LESS,                 bool);
BINARYOP_BENCHMARK_DEFINE(GREATER,        timestamp_ms, timestamp_s,  GREATER,              bool);
BINARYOP_BENCHMARK_DEFINE(NULL_EQUALS,    duration_ms,  duration_ns,  NULL_EQUALS,          bool);
BINARYOP_BENCHMARK_DEFINE(NULL_MAX,       decimal32,    decimal32,    NULL_MAX,             decimal32);
BINARYOP_BENCHMARK_DEFINE(NULL_MIN,       timestamp_D,  timestamp_s,  NULL_MIN,             timestamp_s);
