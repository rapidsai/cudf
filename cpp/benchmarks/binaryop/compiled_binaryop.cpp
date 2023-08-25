/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>

class COMPILED_BINARYOP : public cudf::benchmark {};

template <typename TypeLhs, typename TypeRhs, typename TypeOut>
void BM_compiled_binaryop(benchmark::State& state, cudf::binary_operator binop)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};

  auto const source_table = create_random_table(
    {cudf::type_to_id<TypeLhs>(), cudf::type_to_id<TypeRhs>()}, row_count{column_size});

  auto lhs = cudf::column_view(source_table->get_column(0));
  auto rhs = cudf::column_view(source_table->get_column(1));

  auto output_dtype = cudf::data_type(cudf::type_to_id<TypeOut>());

  // Call once for hot cache.
  cudf::binary_operation(lhs, rhs, binop, output_dtype);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    cudf::binary_operation(lhs, rhs, binop, output_dtype);
  }

  // use number of bytes read and written to global memory
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * column_size *
                          (sizeof(TypeLhs) + sizeof(TypeRhs) + sizeof(TypeOut)));
}

// TODO tparam boolean for null.
#define BM_BINARYOP_BENCHMARK_DEFINE(name, lhs, rhs, bop, tout)           \
  BENCHMARK_DEFINE_F(COMPILED_BINARYOP, name)                             \
  (::benchmark::State & st)                                               \
  {                                                                       \
    BM_compiled_binaryop<lhs, rhs, tout>(st, cudf::binary_operator::bop); \
  }                                                                       \
  BENCHMARK_REGISTER_F(COMPILED_BINARYOP, name)                           \
    ->Unit(benchmark::kMicrosecond)                                       \
    ->UseManualTime()                                                     \
    ->Arg(10000)      /* 10k */                                           \
    ->Arg(100000)     /* 100k */                                          \
    ->Arg(1000000)    /* 1M */                                            \
    ->Arg(10000000)   /* 10M */                                           \
    ->Arg(100000000); /* 100M */

#define build_name(a, b, c, d) a##_##b##_##c##_##d

#define BINARYOP_BENCHMARK_DEFINE(lhs, rhs, bop, tout) \
  BM_BINARYOP_BENCHMARK_DEFINE(build_name(bop, lhs, rhs, tout), lhs, rhs, bop, tout)

using cudf::duration_D;
using cudf::duration_ms;
using cudf::duration_ns;
using cudf::duration_s;
using cudf::timestamp_D;
using cudf::timestamp_ms;
using cudf::timestamp_s;
using numeric::decimal32;

// clang-format off
BINARYOP_BENCHMARK_DEFINE(float,        int64_t,      ADD,                  int32_t);
BINARYOP_BENCHMARK_DEFINE(float,        float,        ADD,                  float);
BINARYOP_BENCHMARK_DEFINE(timestamp_s,  duration_s,   ADD,                  timestamp_s);
BINARYOP_BENCHMARK_DEFINE(duration_s,   duration_D,   SUB,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      SUB,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(float,        float,        MUL,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(duration_s,   int64_t,      MUL,                  duration_s);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      DIV,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  int32_t,      DIV,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      TRUE_DIV,             int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      FLOOR_DIV,            int64_t);
BINARYOP_BENCHMARK_DEFINE(double,       double,       MOD,                  double);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  int64_t,      MOD,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int32_t,      int64_t,      PMOD,                 double);
BINARYOP_BENCHMARK_DEFINE(int32_t,      uint8_t,      PYMOD,                int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      POW,                  double);
BINARYOP_BENCHMARK_DEFINE(float,        double,       LOG_BASE,             double);
BINARYOP_BENCHMARK_DEFINE(float,        double,       ATAN2,                double);
BINARYOP_BENCHMARK_DEFINE(int,          int,          SHIFT_LEFT,           int);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      SHIFT_RIGHT,          int);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int32_t,      SHIFT_RIGHT_UNSIGNED, int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int32_t,      BITWISE_AND,          int16_t);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int32_t,      BITWISE_OR,           int64_t);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      BITWISE_XOR,          int32_t);
BINARYOP_BENCHMARK_DEFINE(double,       int8_t,       LOGICAL_AND,          bool);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      LOGICAL_OR,           bool);
BINARYOP_BENCHMARK_DEFINE(int32_t,      int64_t,      EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  duration_ns,  EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(decimal32,    decimal32,    NOT_EQUAL,            bool);
BINARYOP_BENCHMARK_DEFINE(timestamp_s,  timestamp_s,  LESS,                 bool);
BINARYOP_BENCHMARK_DEFINE(timestamp_ms, timestamp_s,  GREATER,              bool);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  duration_ns,  NULL_EQUALS,          bool);
BINARYOP_BENCHMARK_DEFINE(decimal32,    decimal32,    NULL_MAX,             decimal32);
BINARYOP_BENCHMARK_DEFINE(timestamp_D,  timestamp_s,  NULL_MIN,             timestamp_s);
