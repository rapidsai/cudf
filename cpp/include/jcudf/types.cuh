/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace jcudf {

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

template <typename T>
__device__ constexpr bool bit_is_set(T const* bitmask, usize bit_index)
{
  constexpr auto bits_per_word = sizeof(T) * 8;
  return bitmask[bit_index / bits_per_word] & (T{1} << (bit_index % bits_per_word));
}

template <typename T, typename U>
inline constexpr bool Same = false;

template <typename T>
inline constexpr bool Same<T, T> = true;

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

template <typename R>
struct dec {
  using Rep = R;

  R _value = 0;

  i32 _scale = 0;

  __device__ constexpr dec(scaled_t, R value, i32 scale) : _value{value}, _scale{scale} {}

  constexpr dec() = default;

  __device__ constexpr R value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

using dec32  = dec<i32>;
using dec64  = dec<i64>;
using dec128 = dec<i128>;

template <typename R>
__device__ constexpr auto rescale(dec<R> a, i32 scale)
{
  return dec<R>{scaled, dec_rescale(a._value, a._scale, scale), scale};
}

template <typename R>
__device__ constexpr auto operator+(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value + rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator-(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value - rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr auto operator*(dec<R> a, dec<R> b)
{
  return dec<R>{scaled, a._value * b._value, a._scale + b._scale};
}

template <typename R>
__device__ constexpr auto operator/(dec<R> a, dec<R> b)
{
  return dec<R>{scaled, a._value / b._value, a._scale - b._scale};
}

template <typename R>
__device__ constexpr auto operator%(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  auto r     = rescale(a, scale)._value % rescale(b, scale)._value;
  return dec<R>{scaled, r, scale};
}

template <typename R>
__device__ constexpr int operator<=>(dec<R> a, dec<R> b)
{
  auto scale = min(a._scale, b._scale);
  return rescale(a, scale)._value - rescale(b, scale)._value;
}

enum class timestamp_unit : i32 { D, h, m, s, ms, us, ns };

template <typename R, timestamp_unit Unit>
struct timestamp {
  using Rep = R;

  R _rep = 0;

  __device__ constexpr R count() const { return _rep; }
};

using timestamp_D  = timestamp<i32, timestamp_unit::D>;
using timestamp_h  = timestamp<i32, timestamp_unit::h>;
using timestamp_m  = timestamp<i32, timestamp_unit::m>;
using timestamp_s  = timestamp<i64, timestamp_unit::s>;
using timestamp_ms = timestamp<i64, timestamp_unit::ms>;
using timestamp_us = timestamp<i64, timestamp_unit::us>;
using timestamp_ns = timestamp<i64, timestamp_unit::ns>;

template <typename R, timestamp_unit Unit>
__device__ constexpr int operator<=>(timestamp<R, Unit> a, timestamp<R, Unit> b)
{
  return a._rep - b._rep;
}

template <typename R, timestamp_unit Unit>
struct duration {
  using Rep = R;

  R _rep = 0;

  __device__ constexpr R count() const { return _rep; }
};

using duration_D  = duration<i32, timestamp_unit::D>;
using duration_h  = duration<i32, timestamp_unit::h>;
using duration_m  = duration<i32, timestamp_unit::m>;
using duration_s  = duration<i64, timestamp_unit::s>;
using duration_ms = duration<i64, timestamp_unit::ms>;
using duration_us = duration<i64, timestamp_unit::us>;
using duration_ns = duration<i64, timestamp_unit::ns>;

template <typename R, timestamp_unit Unit>
__device__ constexpr duration<R, Unit> operator+(duration<R, Unit> a, duration<R, Unit> b)
{
  return duration<R, Unit>{a._rep + b._rep};
}

template <typename R, timestamp_unit Unit>
__device__ constexpr duration<R, Unit> operator-(duration<R, Unit> a, duration<R, Unit> b)
{
  return duration<R, Unit>{a._rep - b._rep};
}

template <typename R, timestamp_unit Unit>
__device__ constexpr int operator<=>(duration<R, Unit> a, duration<R, Unit> b)
{
  return a._rep - b._rep;
}

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

__device__ constexpr int operator<=>(string_view const& a, string_view const& b)
{
  return a.compare(b);
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

  template <typename U = T>
  __device__ constexpr T& element(usize idx) const
  {
    return _data[idx];
  }

  __device__ constexpr bool nullable() const { return false; }

  __device__ constexpr bool is_valid_nocheck(usize element_index) const { return true; }

  __device__ constexpr bool is_valid(usize element_index) const { return true; }

  __device__ constexpr bool is_null(usize element_index) const { return false; }

  template <typename U = T>
  __device__ constexpr optional<T> nullable_element(usize idx) const
  {
    if (!is_valid(idx)) return nullopt;
    return element<U>(idx);
  }

  __device__ constexpr void assign(usize idx, T value) const { _data[idx] = value; }
};

template <typename T>
span(T*, usize) -> span<T>;

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

  template <typename U = T>
  __device__ constexpr T& element(usize idx) const
  {
    return _data[idx];
  }

  template <typename U = T>
  __device__ constexpr optional<T> nullable_element(usize idx) const
  {
    if (!is_valid(idx)) return nullopt;
    return element<U>(idx);
  }

  template <typename U = T>
  __device__ constexpr void assign(usize idx, T value) const
  {
    _data[idx] = value;
  }
};

template <typename T>
optional_span(T*, usize, bitmask_t const*) -> optional_span<T>;

struct alignas(16) column_view {
  template <typename T>
  static constexpr bool HasSpanLayout =
    Same<T, bool> || Same<T, i8> || Same<T, i16> || Same<T, i32> || Same<T, i64> || Same<T, u8> ||
    Same<T, u16> || Same<T, u32> || Same<T, u64> || Same<T, f32> || Same<T, f64> ||
    Same<T, timestamp_D> || Same<T, timestamp_h> || Same<T, timestamp_m> || Same<T, timestamp_s> ||
    Same<T, timestamp_ms> || Same<T, timestamp_us> || Same<T, timestamp_ns> ||
    Same<T, duration_D> || Same<T, duration_h> || Same<T, duration_m> || Same<T, duration_s> ||
    Same<T, duration_ms> || Same<T, duration_us> || Same<T, duration_ns>;

  template <typename T>
  static constexpr bool HasDecimalLayout = Same<T, dec32> || Same<T, dec64> || Same<T, dec128>;

  data_type _type = {};

  size_type _size = 0;

  void const* _data = nullptr;

  bitmask_t const* _null_mask = nullptr;

  size_type _offset = 0;

  column_view* _d_children = nullptr;

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
  __device__ auto& element(size_type idx) const
    requires(HasSpanLayout<T>)
  {
    return static_cast<T const*>(_data)[_offset + idx];
  }

  template <typename T>
  __device__ auto element(size_type idx) const
    requires(HasDecimalLayout<T>)
  {
    return T{scaled, static_cast<typename T::Rep const*>(_data)[_offset + idx], _type.scale()};
  }

  template <typename T>
  __device__ string_view element(size_type idx) const
    requires(Same<T, string_view>)
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

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const
  {
    if (!is_valid(idx)) return nullopt;
    return element<T>(idx);
  }
};

// TODO: scope variables should be aligned to avoid uncoalesced reads/writes
namespace scope {

using args = void* const*;

template <int ScopeIndex,
          typename ColumnType /* = column_view, span, optional_span ... */,
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
}  // namespace jcudf
