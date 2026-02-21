/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))

#define JCUDF_EXPORT __attribute__((visibility("default")))

#else

#define JCUDF_EXPORT

#endif

namespace JCUDF_EXPORT jcudf {

using i8   = signed char;
using i16  = signed short;
using i32  = signed int;
using i64  = signed long;
using i128 = __int128_t;
using u8   = unsigned char;
using u16  = unsigned short;
using u32  = unsigned int;
using u64  = unsigned long;

using char_utf8 = u32;

using usize = unsigned long;
using iptr  = i64;
using uptr  = u64;

using intmax_t  = i64;
using uintmax_t = u64;

using f32 = float;
using f64 = double;

using size_type = i32;

using bitmask_t = u32;

__device__ constexpr bool bit_is_set(bitmask_t const* bitmask, usize bit_index)
{
  constexpr auto bits_per_word = sizeof(bitmask_t) * 8;
  return bitmask[bit_index / bits_per_word] & (bitmask_t{1} << (bit_index % bits_per_word));
}

enum class type_id : i32 {
  EMPTY                  = 0,
  INT8                   = 1,
  INT16                  = 2,
  INT32                  = 3,
  INT64                  = 4,
  UINT8                  = 5,
  UINT16                 = 6,
  UINT32                 = 7,
  UINT64                 = 8,
  FLOAT32                = 9,
  FLOAT64                = 10,
  BOOL8                  = 11,
  TIMESTAMP_DAYS         = 12,
  TIMESTAMP_SECONDS      = 13,
  TIMESTAMP_MILLISECONDS = 14,
  TIMESTAMP_MICROSECONDS = 15,
  TIMESTAMP_NANOSECONDS  = 16,
  DURATION_DAYS          = 17,
  DURATION_SECONDS       = 18,
  DURATION_MILLISECONDS  = 19,
  DURATION_MICROSECONDS  = 20,
  DURATION_NANOSECONDS   = 21,
  DICTIONARY32           = 22,
  STRING                 = 23,
  LIST                   = 24,
  DECIMAL32              = 25,
  DECIMAL64              = 26,
  DECIMAL128             = 27,
  STRUCT                 = 28,
  NUM_TYPE_IDS           = 29
};

struct data_type {
  type_id _id = {};

  i32 _scale = 0;

  __device__ constexpr type_id id() const { return _id; }

  __device__ constexpr i32 scale() const { return _scale; }
};

template <typename T>
__device__ constexpr T min(T a, T b)
{
  return a < b ? a : b;
}

template <typename T>
__device__ constexpr T max(T a, T b)
{
  return a > b ? a : b;
}

template <typename T>
__device__ constexpr T ipow10(T exponent)
{
  if (exponent == 0) { return 1; }

  T extra  = 1;
  T square = 10;
  T n      = exponent;

  while (n > 1) {
    if ((n & 1) == 1) { extra *= square; }
    n >>= 1;
    square *= square;
  }

  return square * extra;
}

template <typename T>
__device__ constexpr T dec_lshift(T v, i32 scale)
{
  return v * ipow10(-scale);
}

template <typename T>
__device__ constexpr T dec_rshift(T v, i32 scale)
{
  return v / ipow10(scale);
}

template <typename T>
__device__ constexpr T dec_shift(T v, i32 scale)
{
  if (scale == 0) {
    return v;
  } else if (scale < 0) {
    return dec_lshift(v, scale);
  } else {
    return dec_rshift(v, scale);
  }
}

template <typename T>
__device__ constexpr T dec_rescale(T v, i32 from_scale, i32 to_scale)
{
  return dec_shift(v, to_scale - from_scale);
}

struct scaled_t {};

inline constexpr scaled_t scaled;

struct dec32 {
  i32 _value = 0;

  i32 _scale = 0;

  __device__ constexpr dec32(scaled_t, i32 value, i32 scale) : _value{value}, _scale{scale} {}

  constexpr dec32() = default;

  __device__ constexpr i32 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

struct dec64 {
  i64 _value = 0;

  i32 _scale = 0;

  __device__ constexpr dec64(scaled_t, i64 value, i32 scale) : _value{value}, _scale{scale} {}

  constexpr dec64() = default;

  __device__ constexpr i64 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

struct dec128 {
  i128 _value = 0;

  i32 _scale = 0;

  __device__ constexpr dec128(scaled_t, i128 value, i32 scale) : _value{value}, _scale{scale} {}

  constexpr dec128() = default;

  __device__ constexpr i128 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

#define DECIMAL_OPS(T)                                                               \
  __device__ constexpr T rescale(T a, i32 scale)                                     \
  {                                                                                  \
    return T{scaled, dec_rescale(a._value, a._scale, scale), scale};                 \
  }                                                                                  \
                                                                                     \
  __device__ constexpr T operator+(T const& a, T const& b)                           \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    auto r     = rescale(a, scale)._value + rescale(b, scale)._value;                \
    return T{scaled, r, scale};                                                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr T operator-(T const& a, T const& b)                           \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    auto r     = rescale(a, scale)._value - rescale(b, scale)._value;                \
    return T{scaled, r, scale};                                                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr T operator*(T const& a, T const& b)                           \
  {                                                                                  \
    return T{scaled, a._value + b._value, a._scale + b._scale};                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr T operator/(T const& a, T const& b)                           \
  {                                                                                  \
    return T{scaled, a._value / b._value, a._scale - b._scale};                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr T operator%(T const& a, T const& b)                           \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    auto r     = rescale(a, scale)._value % rescale(b, scale)._value;                \
    return T{scaled, r, scale};                                                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr bool operator==(T const& a, T const& b)                       \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    return rescale(a, scale)._value == rescale(b, scale)._value;                     \
  }                                                                                  \
                                                                                     \
  __device__ constexpr bool operator!=(T const& a, T const& b) { return !(a == b); } \
                                                                                     \
  __device__ constexpr bool operator>(T const& a, T const& b)                        \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    return rescale(a, scale)._value > rescale(b, scale)._value;                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr bool operator<(T const& a, T const& b)                        \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    return rescale(a, scale)._value < rescale(b, scale)._value;                      \
  }                                                                                  \
                                                                                     \
  __device__ constexpr bool operator>=(T const& a, T const& b)                       \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    return rescale(a, scale)._value >= rescale(b, scale)._value;                     \
  }                                                                                  \
                                                                                     \
  __device__ constexpr bool operator<=(T const& a, T const& b)                       \
  {                                                                                  \
    auto scale = min(a._scale, b._scale);                                            \
    return rescale(a, scale)._value <= rescale(b, scale)._value;                     \
  }

DECIMAL_OPS(dec32)
DECIMAL_OPS(dec64)
DECIMAL_OPS(dec128)

struct timestamp_D {
  i32 _rep = 0;
};

struct timestamp_h {
  i32 _rep = 0;
};

struct timestamp_m {
  i32 _rep = 0;
};

struct timestamp_s {
  i64 _rep = 0;
};

struct timestamp_ms {
  i64 _rep = 0;
};

struct timestamp_us {
  i64 _rep = 0;
};

struct timestamp_ns {
  i64 _rep = 0;
};

#define TIMESTAMP_OPS(T)                                                      \
  __device__ constexpr bool operator==(T a, T b) { return a._rep == b._rep; } \
  __device__ constexpr bool operator!=(T a, T b) { return a._rep != b._rep; } \
  __device__ constexpr bool operator>(T a, T b) { return a._rep > b._rep; }   \
  __device__ constexpr bool operator<(T a, T b) { return a._rep < b._rep; }   \
  __device__ constexpr bool operator>=(T a, T b) { return a._rep >= b._rep; } \
  __device__ constexpr bool operator<=(T a, T b) { return a._rep <= b._rep; }

TIMESTAMP_OPS(timestamp_D)
TIMESTAMP_OPS(timestamp_h)
TIMESTAMP_OPS(timestamp_m)
TIMESTAMP_OPS(timestamp_s)
TIMESTAMP_OPS(timestamp_ms)
TIMESTAMP_OPS(timestamp_us)
TIMESTAMP_OPS(timestamp_ns)

struct duration_D {
  i32 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_h {
  i32 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_m {
  i32 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_s {
  i64 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_ms {
  i64 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_us {
  i64 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

struct duration_ns {
  i64 _rep = 0;

  __device__ constexpr auto count() const { return _rep; }
};

#define DURATION_OPS(T)                                                       \
  __device__ constexpr T operator+(T a, T b) { return T{a._rep + b._rep}; }   \
  __device__ constexpr T operator-(T a, T b) { return T{a._rep - b._rep}; }   \
  __device__ constexpr bool operator==(T a, T b) { return a._rep == b._rep; } \
  __device__ constexpr bool operator!=(T a, T b) { return a._rep != b._rep; } \
  __device__ constexpr bool operator>(T a, T b) { return a._rep > b._rep; }   \
  __device__ constexpr bool operator<(T a, T b) { return a._rep < b._rep; }   \
  __device__ constexpr bool operator>=(T a, T b) { return a._rep >= b._rep; } \
  __device__ constexpr bool operator<=(T a, T b) { return a._rep <= b._rep; }

DURATION_OPS(duration_D)
DURATION_OPS(duration_h)
DURATION_OPS(duration_m)
DURATION_OPS(duration_s)
DURATION_OPS(duration_ms)
DURATION_OPS(duration_us)
DURATION_OPS(duration_ns)

struct string_view {
  static constexpr size_type const UNKNOWN_STRING_LENGTH{-1};
  static constexpr size_type const npos{-1};

  char const* _data = nullptr;

  size_type _bytes = 0;

  mutable size_type _length = UNKNOWN_STRING_LENGTH;

  __device__ constexpr size_type size_bytes() const { return _bytes; }

  __device__ constexpr char const* data() const { return _data; }

  __device__ constexpr bool empty() const { return _bytes == 0; }

  __device__ constexpr size_type compare(string_view const& other) const
  {
    auto* s0 = _data;
    auto n0  = _bytes;
    auto* s1 = other._data;
    auto n1  = other._bytes;
    auto max = n0 < n1 ? n0 : n1;

    if (s0 == s1 && n0 == n1) return 0;

    size_type idx = 0;

    while (idx < max) {
      if (*s0 != *s1) return static_cast<i32>(*s0) - static_cast<i32>(*s1);
      s0++;
      s1++;
      idx++;
    }

    if (idx < n0) { return 1; }
    if (idx < n1) { return -1; }

    return 0;
  }
};

__device__ constexpr bool operator==(string_view const& a, string_view const& b)
{
  return a.compare(b) == 0;
}

__device__ constexpr bool operator!=(string_view const& a, string_view const& b)
{
  return a.compare(b) != 0;
}

__device__ constexpr bool operator<(string_view const& a, string_view const& b)
{
  return a.compare(b) < 0;
}

__device__ constexpr bool operator>(string_view const& a, string_view const& b)
{
  return a.compare(b) > 0;
}

__device__ constexpr bool operator<=(string_view const& a, string_view const& b)
{
  return a.compare(b) <= 0;
}

__device__ constexpr bool operator>=(string_view const& a, string_view const& b)
{
  return a.compare(b) >= 0;
}

struct inplace_t {};

inline constexpr inplace_t inplace;

struct nullopt_t {};

inline constexpr nullopt_t nullopt;

template <typename T>
struct optional {
  T _val = {};

  bool _engaged = false;

  constexpr optional() = default;

  __device__ constexpr optional(nullopt_t) {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : _val{static_cast<Args&&>(args)...}, _engaged{true}
  {
  }

  __device__ constexpr optional(T val) : _val{val}, _engaged{true} {}

  __device__ constexpr bool has_value() const { return _engaged; }

  __device__ constexpr bool has_null() const { return !_engaged; }

  __device__ constexpr void reset() { _engaged = false; }

  __device__ constexpr T const& get() const { return _val; }

  __device__ constexpr T& get() { return _val; }

  __device__ constexpr T const* operator->() const { return &_val; }

  __device__ constexpr T* operator->() { return &_val; }

  __device__ constexpr T const& operator*() const { return _val; }

  __device__ constexpr T& operator*() { return _val; }

  __device__ constexpr T const& value() const { return _val; }

  __device__ constexpr T& value() { return _val; }

  __device__ constexpr explicit operator bool() const { return _engaged; }

  __device__ constexpr T value_or(T __v) const { return _engaged ? _val : __v; }
};

template <typename T>
optional(T) -> optional<T>;

#define INST(T) template struct optional<T>;

INST(bool);
INST(i8);
INST(i16);
INST(i32);
INST(i64);
INST(u8);
INST(u16);
INST(u32);
INST(u64);
INST(f32);
INST(f64);
INST(timestamp_D);
INST(timestamp_h);
INST(timestamp_m);
INST(timestamp_s);
INST(timestamp_ms);
INST(timestamp_us);
INST(timestamp_ns);
INST(duration_D);
INST(duration_h);
INST(duration_m);
INST(duration_s);
INST(duration_ms);
INST(duration_us);
INST(duration_ns);
INST(dec32);
INST(dec64);
INST(dec128);
INST(string_view);

#undef INST

template <typename T>
struct span {
  T* _data = nullptr;

  usize _size = 0;

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr usize size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](usize pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr span<T const> as_const() const { return span<T const>{_data, _size}; }

  __device__ constexpr T& element(usize idx) const { return _data[idx]; }

  __device__ constexpr void assign(usize idx, T value) const { _data[idx] = value; }
};

#define INST(T) template struct span<T>;

INST(bool);
INST(i8);
INST(i16);
INST(i32);
INST(i64);
INST(u8);
INST(u16);
INST(u32);
INST(u64);
INST(f32);
INST(f64);
INST(timestamp_D);
INST(timestamp_h);
INST(timestamp_m);
INST(timestamp_s);
INST(timestamp_ms);
INST(timestamp_us);
INST(timestamp_ns);
INST(duration_D);
INST(duration_h);
INST(duration_m);
INST(duration_s);
INST(duration_ms);
INST(duration_us);
INST(duration_ns);
INST(dec32);
INST(dec64);
INST(dec128);
INST(string_view);

#undef INST

template <typename T>
struct optional_span {
  T* _data = nullptr;

  usize _size = 0;

  bitmask_t const* _null_mask = nullptr;

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr usize size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](usize pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr optional_span<T const> as_const() const
  {
    return optional_span<T const>{_data, _size, _null_mask};
  }

  __device__ constexpr bool nullable() const { return _null_mask != nullptr; }

  __device__ constexpr bool is_valid_nocheck(usize element_index) const
  {
    return bit_is_set(_null_mask, element_index);
  }

  __device__ constexpr bool is_valid(usize element_index) const
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  __device__ constexpr bool is_null(usize element_index) const { return !is_valid(element_index); }

  __device__ constexpr T& element(usize idx) const { return _data[idx]; }

  __device__ constexpr optional<T> nullable_element(usize idx) const;

  __device__ constexpr void assign(usize idx, T value) const { _data[idx] = value; }
};

#define INST(T) template struct optional_span<T>;

INST(bool);
INST(i8);
INST(i16);
INST(i32);
INST(i64);
INST(u8);
INST(u16);
INST(u32);
INST(u64);
INST(f32);
INST(f64);
INST(timestamp_D);
INST(timestamp_h);
INST(timestamp_m);
INST(timestamp_s);
INST(timestamp_ms);
INST(timestamp_us);
INST(timestamp_ns);
INST(duration_D);
INST(duration_h);
INST(duration_m);
INST(duration_s);
INST(duration_ms);
INST(duration_us);
INST(duration_ns);
INST(dec32);
INST(dec64);
INST(dec128);
INST(string_view);

#undef INST

struct alignas(16) column_device_view {
  data_type _type = {};

  size_type _size = 0;

  void const* _data = nullptr;

  bitmask_t const* _null_mask = nullptr;

  size_type _offset = 0;

  column_device_view* _d_children = nullptr;

  size_type _num_children = 0;

  __device__ constexpr data_type type() const { return _type; }

  __device__ constexpr size_type size() const { return _size; }

  __device__ constexpr bool nullable() const { return _null_mask != nullptr; }

  __device__ constexpr bitmask_t const* null_mask() const { return _null_mask; }

  __device__ constexpr size_type offset() const { return _offset; }

  __device__ constexpr bool is_valid(size_type idx) const
  {
    return !nullable() || is_valid_nocheck(idx);
  }

  __device__ constexpr bool is_valid_nocheck(size_type idx) const
  {
    return bit_is_set(_null_mask, _offset + idx);
  }

  __device__ constexpr bool is_null(size_type idx) const { return !is_valid(idx); }

  __device__ constexpr bool is_null_nocheck(size_type idx) const { return !is_valid_nocheck(idx); }

  __device__ constexpr size_type num_child_columns() const { return _num_children; }

  template <typename T>
  __device__ T element(size_type idx) const;

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const
  {
    if (!is_valid(idx)) return nullopt;
    return element<T>(idx);
  }
};

#define SPEC(T)                                                    \
  template <>                                                      \
  __device__ T column_device_view::element<T>(size_type idx) const \
  {                                                                \
    return static_cast<T const*>(_data)[_offset + idx];            \
  }

SPEC(bool)
SPEC(i8)
SPEC(i16)
SPEC(i32)
SPEC(i64)
SPEC(u8)
SPEC(u16)
SPEC(u32)
SPEC(u64)
SPEC(f32)
SPEC(f64)
SPEC(timestamp_D)
SPEC(timestamp_h)
SPEC(timestamp_m)
SPEC(timestamp_s)
SPEC(timestamp_ms)
SPEC(timestamp_us)
SPEC(timestamp_ns)
SPEC(duration_D)
SPEC(duration_h)
SPEC(duration_m)
SPEC(duration_s)
SPEC(duration_ms)
SPEC(duration_us)
SPEC(duration_ns)

#undef SPEC

#define SPEC(T, Repr)                                                                \
  template <>                                                                        \
  __device__ T column_device_view::element<T>(size_type idx) const                   \
  {                                                                                  \
    return T{scaled, static_cast<Repr const*>(_data)[_offset + idx], _type.scale()}; \
  }

SPEC(dec32, i32)
SPEC(dec64, i64)
SPEC(dec128, i128)

#undef SPEC

template <>
__device__ string_view column_device_view::element<string_view>(size_type idx) const
{
  static constexpr i32 OFFSETS_CHILD = 0;
  auto i                             = _offset + idx;
  auto* str_data                     = static_cast<char const*>(_data);
  auto& offsets                      = _d_children[OFFSETS_CHILD];
  auto* i32_runs                     = static_cast<i32 const*>(offsets._data);
  auto* i64_runs                     = static_cast<i64 const*>(offsets._data);

  i64 run_begin = 0;
  i64 run_end   = 0;

  switch (offsets.type().id()) {
    case type_id::INT32:
      run_begin = i32_runs[i];
      run_end   = i32_runs[i + 1];
      break;
    case type_id::INT64:
      run_begin = i64_runs[i];
      run_end   = i64_runs[i + 1];
      break;
    default: __builtin_unreachable();
  }

  i64 run_size = run_end - run_begin;

  return string_view{str_data + run_begin, static_cast<size_type>(run_size)};
}

#define INST(T) \
  template __device__ optional<T> column_device_view::nullable_element<T>(size_type idx) const;

INST(bool)
INST(i8)
INST(i16)
INST(i32)
INST(i64)
INST(u8)
INST(u16)
INST(u32)
INST(u64)
INST(f32)
INST(f64)
INST(timestamp_D)
INST(timestamp_h)
INST(timestamp_m)
INST(timestamp_s)
INST(timestamp_ms)
INST(timestamp_us)
INST(timestamp_ns)
INST(duration_D)
INST(duration_h)
INST(duration_m)
INST(duration_s)
INST(duration_ms)
INST(duration_us)
INST(duration_ns)
INST(dec32)
INST(dec64)
INST(dec128)
INST(string_view)

#undef INST

/// @brief Type-erased parameters for LTO-JIT-compiled transform operations.
struct transform_operator_params {
  /// @brief Pointer to scope data (e.g. column views, scalars, etc.).
  void* const* scope = nullptr;

  /// @brief Total number of rows to process.
  size_type num_rows = 0;

  /// @brief Current row index.
  size_type row_index = 0;
};

// TODO: scope variables should be aligned to avoid uncoalesced reads/writes
namespace scope {

using args = void* const*;

template <int ScopeIndex,
          typename ColumnType /* = column_device_view, mutable_column_device_view, span,
                                 optional_span ... */
          ,
          typename T /* = int, float, fixed_point, string_view ... */,
          bool IsScalar,
          bool IsNullable>
struct column {
  static constexpr bool IS_SCALAR   = IsScalar;
  static constexpr bool IS_NULLABLE = IsNullable;

  using Type = T;
  using Arg  = ColumnType const*;

  static __device__ decltype(auto) element(args scope, size_type i)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    if constexpr (!IsNullable) {
      return p->template element<T>(index);
    } else {
      return p->template nullable_element<T>(index);
    }
  }

  static __device__ void assign(args scope, size_type i, T value)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    p->template assign<T>(index, value);
  }

  static __device__ auto* null_mask(args scope)
  {
    auto p = static_cast<Arg>(scope[ScopeIndex]);
    return p->null_mask();
  }

  static __device__ bool is_null(args scope, size_type i)
  {
    if constexpr (!IsNullable) { return false; }

    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    return p->is_null(index);
  }

  static __device__ bool is_valid(args scope, size_type i) { return !is_null(scope, i); }
};

template <int ScopeIndex>
struct user_data {
  using Arg = void*;

  static __device__ decltype(auto) element(args scope, [[maybe_unused]] size_type i)
  {
    return static_cast<Arg>(scope[ScopeIndex]);
  }
};

}  // namespace scope
}  // namespace JCUDF_EXPORT jcudf
