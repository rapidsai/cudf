/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))

#define CUDF_LTO_EXPORT __attribute__((visibility("default")))

#else

#define CUDF_LTO_EXPORT

#endif

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

using i8   = signed char;
using i16  = signed short;
using i32  = signed int;
using i64  = signed long;
using i128 = __int128_t;
using u8   = unsigned char;
using u16  = unsigned short;
using u32  = unsigned int;
using u64  = unsigned long;

using size_t = unsigned long;
using iptr   = i64;
using uptr   = u64;

using intmax_t  = i64;
using uintmax_t = u64;

using f32 = float;
using f64 = double;

using size_type = i32;

using bitmask_type = u32;

__device__ constexpr bool bit_is_set(bitmask_type const* bitmask, size_t bit_index)
{
  constexpr auto bits_per_word = sizeof(bitmask_type) * 8;
  return bitmask[bit_index / bits_per_word] & (bitmask_type{1} << (bit_index % bits_per_word));
}

using char_utf8 = u32;

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

struct scaled_t {};

inline constexpr scaled_t scaled{};

struct decimal32 {
  i32 _value = 0;

  i32 _scale = 0;

  __device__ constexpr decimal32(scaled_t, i32 value, i32 scale) : _value{value}, _scale{scale} {}

  __device__ constexpr i32 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

struct decimal64 {
  i64 _value = 0;

  i32 _scale = 0;

  __device__ constexpr decimal64(scaled_t, i64 value, i32 scale) : _value{value}, _scale{scale} {}

  __device__ constexpr i64 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

struct decimal128 {
  i128 _value = 0;

  i32 _scale = 0;

  __device__ constexpr decimal128(scaled_t, i128 value, i32 scale) : _value{value}, _scale{scale} {}

  __device__ constexpr i128 value() const { return _value; }

  __device__ constexpr i32 scale() const { return _scale; }
};

// TODO: implement
#define DECIMAL_OPS(T)                                        \
  __device__ extern T operator+(T const& lhs, T const& rhs);  \
  __device__ extern T operator-(T const& lhs, T const& rhs);  \
  __device__ extern T operator*(T const& lhs, T const& rhs);  \
  __device__ extern T operator/(T const& lhs, T const& rhs);  \
  __device__ extern T operator==(T const& lhs, T const& rhs); \
  __device__ extern T operator!=(T const& lhs, T const& rhs); \
  __device__ extern T operator>(T const& lhs, T const& rhs);  \
  __device__ extern T operator<(T const& lhs, T const& rhs);  \
  __device__ extern T operator>=(T const& lhs, T const& rhs); \
  __device__ extern T operator<=(T const& lhs, T const& rhs);

DECIMAL_OPS(decimal32)
DECIMAL_OPS(decimal64)
DECIMAL_OPS(decimal128)

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

#define TIMESTAMP_OPS(T)                                                                         \
  __device__ constexpr T operator==(T const& lhs, T const& rhs) { return lhs._rep == rhs._rep; } \
  __device__ constexpr T operator!=(T const& lhs, T const& rhs) { return lhs._rep != rhs._rep; } \
  __device__ constexpr T operator>(T const& lhs, T const& rhs) { return lhs._rep > rhs._rep; }   \
  __device__ constexpr T operator<(T const& lhs, T const& rhs) { return lhs._rep < rhs._rep; }   \
  __device__ constexpr T operator>=(T const& lhs, T const& rhs) { return lhs._rep >= rhs._rep; } \
  __device__ constexpr T operator<=(T const& lhs, T const& rhs) { return lhs._rep <= rhs._rep; }

TIMESTAMP_OPS(timestamp_D)
TIMESTAMP_OPS(timestamp_h)
TIMESTAMP_OPS(timestamp_m)
TIMESTAMP_OPS(timestamp_s)
TIMESTAMP_OPS(timestamp_ms)
TIMESTAMP_OPS(timestamp_us)
TIMESTAMP_OPS(timestamp_ns)

struct duration_D {
  i32 _rep = 0;
};

struct duration_h {
  i32 _rep = 0;
};

struct duration_m {
  i32 _rep = 0;
};

struct duration_s {
  i64 _rep = 0;
};

struct duration_ms {
  i64 _rep = 0;
};

struct duration_us {
  i64 _rep = 0;
};

struct duration_ns {
  i64 _rep = 0;
};

#define DURATION_OPS(T)                                                                           \
  __device__ constexpr T operator+(T const& lhs, T const& rhs) { return T{lhs._rep + rhs._rep}; } \
  __device__ constexpr T operator-(T const& lhs, T const& rhs) { return T{lhs._rep - rhs._rep}; } \
  __device__ constexpr T operator==(T const& lhs, T const& rhs) { return lhs._rep == rhs._rep; }  \
  __device__ constexpr T operator!=(T const& lhs, T const& rhs) { return lhs._rep != rhs._rep; }  \
  __device__ constexpr T operator>(T const& lhs, T const& rhs) { return lhs._rep > rhs._rep; }    \
  __device__ constexpr T operator<(T const& lhs, T const& rhs) { return lhs._rep < rhs._rep; }    \
  __device__ constexpr T operator>=(T const& lhs, T const& rhs) { return lhs._rep >= rhs._rep; }  \
  __device__ constexpr T operator<=(T const& lhs, T const& rhs) { return lhs._rep <= rhs._rep; }

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

  char const* _data         = nullptr;
  size_type _bytes          = 0;
  mutable size_type _length = UNKNOWN_STRING_LENGTH;

  __device__ size_type size_bytes() const { return _bytes; }

  __device__ char const* data() const { return _data; }

  __device__ bool empty() const { return _bytes == 0; }
};

struct inplace_t {};

inline constexpr inplace_t inplace{};

struct nullopt_t {};

inline constexpr nullopt_t nullopt{};

// TODO: assumes T is trivially copyable
template <typename T>
struct optional {
  T _val;
  bool _engaged;

  __device__ constexpr optional() : _val{}, _engaged{false} {}

  __device__ constexpr optional(nullopt_t) : _val{}, _engaged{false} {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : _val{static_cast<Args&&>(args)...}, _engaged{true}
  {
  }

  __device__ constexpr optional(T val) : _val{val}, _engaged{true} {}

  constexpr optional(optional const&) = default;

  constexpr optional(optional&&) = default;

  constexpr optional& operator=(optional const&) = default;

  constexpr optional& operator=(optional&&) = default;

  constexpr ~optional() = default;

  __device__ constexpr bool has_value() const { return _engaged; }

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

template struct optional<bool>;
template struct optional<i8>;
template struct optional<i16>;
template struct optional<i32>;
template struct optional<i64>;
template struct optional<u8>;
template struct optional<u16>;
template struct optional<u32>;
template struct optional<u64>;
template struct optional<f32>;
template struct optional<f64>;
template struct optional<timestamp_D>;
template struct optional<timestamp_h>;
template struct optional<timestamp_m>;
template struct optional<timestamp_s>;
template struct optional<timestamp_ms>;
template struct optional<timestamp_us>;
template struct optional<timestamp_ns>;
template struct optional<duration_D>;
template struct optional<duration_h>;
template struct optional<duration_m>;
template struct optional<duration_s>;
template struct optional<duration_ms>;
template struct optional<duration_us>;
template struct optional<duration_ns>;
template struct optional<decimal32>;
template struct optional<decimal64>;
template struct optional<decimal128>;
template struct optional<string_view>;

template <typename T>
struct span {
  T* _data = nullptr;

  size_t _size = 0;

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr size_t size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](size_t pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr span<T const> as_const() const { return span<T const>{_data, _size}; }

  __device__ constexpr T& element(size_t idx) const { return _data[idx]; }

  __device__ constexpr void assign(size_t idx, T value) const { _data[idx] = value; }
};

template struct span<bool>;
template struct span<i8>;
template struct span<i16>;
template struct span<i32>;
template struct span<i64>;
template struct span<u8>;
template struct span<u16>;
template struct span<u32>;
template struct span<u64>;
template struct span<f32>;
template struct span<f64>;
template struct span<timestamp_D>;
template struct span<timestamp_h>;
template struct span<timestamp_m>;
template struct span<timestamp_s>;
template struct span<timestamp_ms>;
template struct span<timestamp_us>;
template struct span<timestamp_ns>;
template struct span<duration_D>;
template struct span<duration_h>;
template struct span<duration_m>;
template struct span<duration_s>;
template struct span<duration_ms>;
template struct span<duration_us>;
template struct span<duration_ns>;
template struct span<decimal32>;
template struct span<decimal64>;
template struct span<decimal128>;
template struct span<string_view>;

template <typename T>
struct optional_span {
  T* _data = nullptr;

  size_t _size = 0;

  bitmask_type const* _null_mask = nullptr;

  __device__ constexpr T* data() const { return _data; }

  __device__ constexpr size_t size() const { return _size; }

  __device__ constexpr bool empty() const { return _size == 0; }

  __device__ constexpr T& operator[](size_t pos) const { return _data[pos]; }

  __device__ constexpr T* begin() const { return _data; }

  __device__ constexpr T* end() const { return _data + _size; }

  __device__ constexpr optional_span<T const> as_const() const
  {
    return optional_span<T const>{_data, _size, _null_mask};
  }

  __device__ constexpr bool nullable() const { return _null_mask != nullptr; }

  __device__ constexpr bool is_valid_nocheck(size_t element_index) const
  {
    return bit_is_set(_null_mask, element_index);
  }

  __device__ constexpr bool is_valid(size_t element_index) const
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  __device__ constexpr bool is_null(size_t element_index) const { return !is_valid(element_index); }

  __device__ constexpr T& element(size_t idx) const { return _data[idx]; }

  __device__ constexpr optional<T> nullable_element(size_t idx) const;

  __device__ constexpr void assign(size_t idx, T value) const { _data[idx] = value; }
};

template struct optional_span<bool>;
template struct optional_span<i8>;
template struct optional_span<i16>;
template struct optional_span<i32>;
template struct optional_span<i64>;
template struct optional_span<u8>;
template struct optional_span<u16>;
template struct optional_span<u32>;
template struct optional_span<u64>;
template struct optional_span<f32>;
template struct optional_span<f64>;
template struct optional_span<timestamp_D>;
template struct optional_span<timestamp_h>;
template struct optional_span<timestamp_m>;
template struct optional_span<timestamp_s>;
template struct optional_span<timestamp_ms>;
template struct optional_span<timestamp_us>;
template struct optional_span<timestamp_ns>;
template struct optional_span<duration_D>;
template struct optional_span<duration_h>;
template struct optional_span<duration_m>;
template struct optional_span<duration_s>;
template struct optional_span<duration_ms>;
template struct optional_span<duration_us>;
template struct optional_span<duration_ns>;
template struct optional_span<decimal32>;
template struct optional_span<decimal64>;
template struct optional_span<decimal128>;
template struct optional_span<string_view>;

struct alignas(16) column_device_view {
  data_type _type = {};

  size_type _size = 0;

  void const* _data = nullptr;

  bitmask_type const* _null_mask = nullptr;

  size_type _offset = 0;

  column_device_view* _d_children = nullptr;

  size_type _num_children = 0;

  __device__ constexpr size_type size() const { return _size; }

  __device__ constexpr bool nullable() const { return _null_mask != nullptr; }

  __device__ constexpr bitmask_type const* null_mask() const { return _null_mask; }

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

  __device__ constexpr bitmask_type get_mask_word(size_type word_index) const
  {
    return _null_mask[word_index];
  }

  __device__ constexpr size_type num_child_columns() const { return _num_children; }

  template <typename T>
  __device__ constexpr T element(size_type idx) const;

  template <typename T>
  __device__ constexpr optional<T> nullable_element(size_type idx) const
  {
    if (!is_valid(idx)) return nullopt;
    return element<T>(idx);
  }
};

#define CUDF_SPEC(T)                                                         \
  template <>                                                                \
  __device__ constexpr T column_device_view::element<T>(size_type idx) const \
  {                                                                          \
    return static_cast<T const*>(_data)[_offset + idx];                      \
  }

CUDF_SPEC(bool)
CUDF_SPEC(i8)
CUDF_SPEC(i16)
CUDF_SPEC(i32)
CUDF_SPEC(i64)
CUDF_SPEC(u8)
CUDF_SPEC(u16)
CUDF_SPEC(u32)
CUDF_SPEC(u64)
CUDF_SPEC(f32)
CUDF_SPEC(f64)
CUDF_SPEC(timestamp_D)
CUDF_SPEC(timestamp_h)
CUDF_SPEC(timestamp_m)
CUDF_SPEC(timestamp_s)
CUDF_SPEC(timestamp_ms)
CUDF_SPEC(timestamp_us)
CUDF_SPEC(timestamp_ns)
CUDF_SPEC(duration_D)
CUDF_SPEC(duration_h)
CUDF_SPEC(duration_m)
CUDF_SPEC(duration_s)
CUDF_SPEC(duration_ms)
CUDF_SPEC(duration_us)
CUDF_SPEC(duration_ns)

#undef CUDF_SPEC

#define CUDF_SPEC(T, Repr)                                                   \
  template <>                                                                \
  __device__ constexpr T column_device_view::element<T>(size_type idx) const \
  {                                                                          \
    return T{static_cast<Repr const*>(_data)[_offset + idx], _type.scale()}; \
  }

CUDF_SPEC(decimal32, i32)
CUDF_SPEC(decimal64, i64)
CUDF_SPEC(decimal128, i128)

#undef CUDF_SPEC

template <>
__device__ constexpr string_view column_device_view::element<string_view>(size_type idx) const
{
  inline constexpr i32 OFFSETS_CHILD = 0;
  auto i                             = _offset + idx;
  auto* str_data                     = static_cast<char const*>(_data);
  auto& offsets                      = _d_children[OFFSETS_CHILD];
  auto* i32_offsets                  = static_cast<i32 const*>(offsets._data);
  auto* i64_offsets                  = static_cast<i64 const*>(offsets._data);

  i64 run_begin = 0;
  i64 run_end   = 0;

  if (offsets.type().id() == type_id::INT32) {
    run_begin = i32_offsets[i];
    run_end   = i32_offsets[i + 1];
  } else {
    run_begin = i64_offsets[i];
    run_end   = i64_offsets[i + 1];
  }

  return string_view{str_data + run_begin, static_cast<size_type>(run_end - run_begin)};
}

#define CUDF_INST(T)
template __device__ constexpr optional<T> column_device_view::nullable_element<T>(
  size_type idx) const;

CUDF_INST(bool)
CUDF_INST(i8)
CUDF_INST(i16)
CUDF_INST(i32)
CUDF_INST(i64)
CUDF_INST(u8)
CUDF_INST(u16)
CUDF_INST(u32)
CUDF_INST(u64)
CUDF_INST(f32)
CUDF_INST(f64)
CUDF_INST(timestamp_D)
CUDF_INST(timestamp_h)
CUDF_INST(timestamp_m)
CUDF_INST(timestamp_s)
CUDF_INST(timestamp_ms)
CUDF_INST(timestamp_us)
CUDF_INST(timestamp_ns)
CUDF_INST(duration_D)
CUDF_INST(duration_h)
CUDF_INST(duration_m)
CUDF_INST(duration_s)
CUDF_INST(duration_ms)
CUDF_INST(duration_us)
CUDF_INST(duration_ns)
CUDF_INST(decimal32)
CUDF_INST(decimal64)
CUDF_INST(decimal128)
CUDF_INST(string_view)

#undef CUDF_INST

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

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
