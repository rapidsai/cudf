/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/export.cuh>

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

// TODO: update doc, and specify this is our ABI for JIT code and should be kept different from
// normal code
/**
 * @brief LTO-JIT functions and thunk types
 *
 * These are declarations for functions that will be used in LTO-JIT compiled code.
 * They are pre-compiled into a device library that is linked at JIT compile time.
 * This header should be minimal and only contain necessary types and function declarations as it
 * will be included and compiled at JIT compile time. Including other headers will lead to longer
 * JIT compile times which can be unbounded and cause slowdowns.
 *
 * This essentially serves as the ABI for LTO-JIT compiled code to interact with the rest of cuDF.
 * Any changes to this header should be made with ABI stability in mind as it can break existing
 * LTO-JIT compiled code and lead to undefined behavior. For example, adding new member variables to
 * these structs will change their size and layout which can break existing code. Adding new
 * functions is generally safe as long as they don't change the existing function signatures, but it
 * can still lead to issues if the new functions are called from existing code that wasn't compiled
 * with them. Removing or changing existing functions is not safe and will break existing code.
 * Changing the types of existing member variables can also break existing code if it changes the
 * size or layout of the structs. In general, any change to this header should be made with caution
 * and thorough testing to ensure ABI compatibility.
 *
 */

using int8_t   = signed char;
using int16_t  = signed short;
using int32_t  = signed int;
using int64_t  = signed long long;
using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

using size_t    = unsigned long;
using intptr_t  = int64_t;
using uintptr_t = uint64_t;

using intmax_t  = int64_t;
using uintmax_t = uint64_t;

using float32_t = float;
using float64_t = double;

using size_type = int32_t;

using bitmask_type = uint32_t;

using char_utf8 = uint32_t;

enum class type_id : int32_t {};

enum scale_type : int32_t {};

struct CUDF_LTO_ALIAS data_type {
 private:
  type_id _id                = {};
  int32_t _fixed_point_scale = 0;
};

struct CUDF_LTO_ALIAS decimal32 {
 private:
  int32_t _value    = 0;
  scale_type _scale = scale_type{};
};

struct CUDF_LTO_ALIAS decimal64 {
 private:
  int64_t _value    = 0;
  scale_type _scale = scale_type{};
};

struct CUDF_LTO_ALIAS decimal128 {
 private:
  __int128_t _value = 0;
  scale_type _scale = scale_type{};
};

struct CUDF_LTO_ALIAS timestamp_D {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_h {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_m {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_s {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_ms {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_us {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_ns {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_D {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_h {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_m {
 private:
  int32_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_s {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_ms {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_us {
 private:
  int64_t _rep = 0;
};

struct CUDF_LTO_ALIAS duration_ns {
 private:
  int64_t _rep = 0;
};

template <typename T>
struct CUDF_LTO_ALIAS optional;

struct CUDF_LTO_ALIAS string_view;

struct CUDF_LTO_ALIAS column_view;

struct CUDF_LTO_ALIAS mutable_column_view;

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
