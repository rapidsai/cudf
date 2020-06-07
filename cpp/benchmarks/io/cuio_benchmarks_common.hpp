/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/io/types.hpp>

// used to make CUIO_BENCH_ALL_TYPES calls more readable
constexpr int UNCOMPRESSED = (int)cudf::io::compression_type::NONE;
constexpr int USE_SNAPPY   = (int)cudf::io::compression_type::SNAPPY;

#define CUIO_BENCH_ALL_TYPES(benchmark_define, compression)                         \
  benchmark_define(Boolean##_##compression, bool, compression);                     \
  benchmark_define(Byte##_##compression, int8_t, compression);                      \
  benchmark_define(Ubyte##_##compression, uint8_t, compression);                    \
  benchmark_define(Short##_##compression, int16_t, compression);                    \
  benchmark_define(Ushort##_##compression, uint16_t, compression);                  \
  benchmark_define(Int##_##compression, int32_t, compression);                      \
  benchmark_define(Uint##_##compression, uint32_t, compression);                    \
  benchmark_define(Long##_##compression, int64_t, compression);                     \
  benchmark_define(Ulong##_##compression, uint64_t, compression);                   \
  benchmark_define(Float##_##compression, float, compression);                      \
  benchmark_define(Double##_##compression, double, compression);                    \
  benchmark_define(String##_##compression, std::string, compression);               \
  benchmark_define(Timestamp_days##_##compression, cudf::timestamp_D, compression); \
  benchmark_define(Timestamp_sec##_##compression, cudf::timestamp_s, compression);  \
  benchmark_define(Timestamp_ms##_##compression, cudf::timestamp_ms, compression);  \
  benchmark_define(Timestamp_us##_##compression, cudf::timestamp_us, compression);  \
  benchmark_define(Timestamp_ns##_##compression, cudf::timestamp_ns, compression);

// sample benchmark define macro that can be passed to the macro above
#define SAMPLE_BENCHMARK_DEFINE(name, datatype, compression)             \
  BENCHMARK_TEMPLATE_DEFINE_F(SampleFixture, name, datatype)             \
  (::benchmark::State & state) { SampleBenchFunction<datatype>(state); } \
  BENCHMARK_REGISTER_F(SampleFixture, name)                              \
    ->Args({data_size, 64, compression})                                 \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseManualTime();

// sample CUIO_BENCH_ALL_TYPES use
// CUIO_BENCH_ALL_TYPES(SAMPLE_BENCHMARK_DEFINE, USE_SNAPPY)
