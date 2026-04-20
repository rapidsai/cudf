/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace __attribute__((visibility("hidden"))) cudf
{
  namespace lite {

  using int8_t   = signed char;
  using int16_t  = signed short;
  using int32_t  = signed int;
  using int64_t  = signed long long;
  using int128_t = __int128_t;

  using uint8_t   = unsigned char;
  using uint16_t  = unsigned short;
  using uint32_t  = unsigned int;
  using uint64_t  = unsigned long long;
  using uint128_t = __uint128_t;

  using size_t    = unsigned long long;
  using intptr_t  = int64_t;
  using uintptr_t = uint64_t;

  using intmax_t  = int64_t;
  using uintmax_t = uint64_t;

  using float32_t = float;
  using float64_t = double;

  using size_type = int32_t;

  using char_utf8 = uint32_t;

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

  struct scaled_t {};

  inline constexpr scaled_t scaled;

  template <typename R>
  struct decimal {
    using Rep = R;

    R _value = 0;

    int32_t _scale = 0;

    constexpr decimal() = default;

    __device__ constexpr decimal(scaled_t, R value, int32_t scale) : _value{value}, _scale{scale} {}

    __device__ constexpr R value() const { return _value; }

    __device__ constexpr int32_t scale() const { return _scale; }

   private:
    __device__ static constexpr R _lshift(R v, int32_t scale) { return v * ipow10(-scale); }

    __device__ static constexpr R _rshift(R v, int32_t scale) { return v / ipow10(scale); }

    __device__ static constexpr R _shift(R v, int32_t scale)
    {
      if (scale == 0) {
        return v;
      } else if (scale < 0) {
        return _lshift(v, scale);
      } else {
        return _rshift(v, scale);
      }
    }

    __device__ static constexpr R _rescale(R v, int32_t from_scale, int32_t to_scale)
    {
      return _shift(v, to_scale - from_scale);
    }

   public:
    __device__ constexpr auto rescale(int32_t scale) const
    {
      return decimal{scaled, _rescale(_value, _scale, scale), scale};
    }

    __device__ constexpr auto operator+(decimal rhs) const
    {
      auto scale = min(_scale, rhs._scale);
      auto r     = rescale(scale)._value + rhs.rescale(scale)._value;
      return decimal{scaled, r, scale};
    }

    __device__ constexpr auto operator-(decimal rhs) const
    {
      auto scale = min(_scale, rhs._scale);
      auto r     = rescale(scale)._value - rhs.rescale(scale)._value;
      return decimal{scaled, r, scale};
    }

    __device__ constexpr auto operator*(decimal rhs) const
    {
      return decimal{scaled, _value * rhs._value, _scale + rhs._scale};
    }

    __device__ constexpr auto operator/(decimal rhs) const
    {
      return decimal{scaled, _value / rhs._value, _scale - rhs._scale};
    }

    __device__ constexpr auto operator%(decimal rhs) const
    {
      auto scale = min(_scale, rhs._scale);
      auto r     = rescale(scale)._value % rhs.rescale(scale)._value;
      return decimal{scaled, r, scale};
    }

    __device__ constexpr int compare(decimal rhs) const
    {
      auto scale = min(_scale, rhs._scale);
      return rescale(scale)._value - rhs.rescale(scale)._value;
    }

    __device__ constexpr bool operator==(decimal rhs) const { return compare(rhs) == 0; }
    __device__ constexpr bool operator!=(decimal rhs) const { return compare(rhs) != 0; }
    __device__ constexpr bool operator<(decimal rhs) const { return compare(rhs) < 0; }
    __device__ constexpr bool operator<=(decimal rhs) const { return compare(rhs) <= 0; }
    __device__ constexpr bool operator>(decimal rhs) const { return compare(rhs) > 0; }
    __device__ constexpr bool operator>=(decimal rhs) const { return compare(rhs) >= 0; }
  };

  using decimal32  = decimal<int32_t>;
  using decimal64  = decimal<int64_t>;
  using decimal128 = decimal<int128_t>;

  enum class timestamp_unit : int32_t { D, h, m, s, ms, us, ns };

  template <typename R, timestamp_unit Unit>
  struct timestamp {
    using Rep = R;

    R _rep = 0;

    __device__ constexpr R count() const { return _rep; }

    __device__ constexpr int compare(timestamp rhs) const { return _rep - rhs._rep; }

    __device__ constexpr bool operator==(timestamp rhs) const { return compare(rhs) == 0; }
    __device__ constexpr bool operator!=(timestamp rhs) const { return compare(rhs) != 0; }
    __device__ constexpr bool operator<(timestamp rhs) const { return compare(rhs) < 0; }
    __device__ constexpr bool operator<=(timestamp rhs) const { return compare(rhs) <= 0; }
    __device__ constexpr bool operator>(timestamp rhs) const { return compare(rhs) > 0; }
    __device__ constexpr bool operator>=(timestamp rhs) const { return compare(rhs) >= 0; }
  };

  using timestamp_D  = timestamp<int32_t, timestamp_unit::D>;
  using timestamp_h  = timestamp<int32_t, timestamp_unit::h>;
  using timestamp_m  = timestamp<int32_t, timestamp_unit::m>;
  using timestamp_s  = timestamp<int64_t, timestamp_unit::s>;
  using timestamp_ms = timestamp<int64_t, timestamp_unit::ms>;
  using timestamp_us = timestamp<int64_t, timestamp_unit::us>;
  using timestamp_ns = timestamp<int64_t, timestamp_unit::ns>;

  template <typename R, timestamp_unit Unit>
  struct duration {
    using Rep = R;

    R _rep = 0;

    __device__ constexpr R count() const { return _rep; }

    __device__ constexpr duration operator+(duration rhs) const
    {
      return duration{_rep + rhs._rep};
    }

    __device__ constexpr duration operator-(duration rhs) const
    {
      return duration{_rep - rhs._rep};
    }

    __device__ constexpr int compare(duration rhs) const { return _rep - rhs._rep; }

    __device__ constexpr bool operator==(duration rhs) const { return compare(rhs) == 0; }
    __device__ constexpr bool operator!=(duration rhs) const { return compare(rhs) != 0; }
    __device__ constexpr bool operator<(duration rhs) const { return compare(rhs) < 0; }
    __device__ constexpr bool operator<=(duration rhs) const { return compare(rhs) <= 0; }
    __device__ constexpr bool operator>(duration rhs) const { return compare(rhs) > 0; }
    __device__ constexpr bool operator>=(duration rhs) const { return compare(rhs) >= 0; }
  };

  using duration_D  = duration<int32_t, timestamp_unit::D>;
  using duration_h  = duration<int32_t, timestamp_unit::h>;
  using duration_m  = duration<int32_t, timestamp_unit::m>;
  using duration_s  = duration<int64_t, timestamp_unit::s>;
  using duration_ms = duration<int64_t, timestamp_unit::ms>;
  using duration_us = duration<int64_t, timestamp_unit::us>;
  using duration_ns = duration<int64_t, timestamp_unit::ns>;

  struct inplace_t {};

  inline constexpr inplace_t inplace;

  struct nullopt_t {};

  inline constexpr nullopt_t nullopt;

  template <typename T>
  struct optional {
    T _value = {};

    bool _is_valid = false;

    constexpr optional() = default;

    __device__ constexpr optional(nullopt_t) {}

    template <typename... Args>
    __device__ constexpr optional(inplace_t, Args&&... args)
      : _value{static_cast<Args&&>(args)...}, _is_valid{true}
    {
    }

    __device__ constexpr optional(T value) : _value{value}, _is_valid{true} {}

    __device__ constexpr bool is_valid() const { return _is_valid; }

    __device__ constexpr bool is_null() const { return !_is_valid; }

    __device__ constexpr void reset() { _is_valid = false; }

    __device__ constexpr T const& get() const { return _value; }

    __device__ constexpr T& get() { return _value; }

    __device__ constexpr T const* operator->() const { return &_value; }

    __device__ constexpr T* operator->() { return &_value; }

    __device__ constexpr T const& operator*() const { return _value; }

    __device__ constexpr T& operator*() { return _value; }

    __device__ constexpr T const& value() const { return _value; }

    __device__ constexpr T& value() { return _value; }

    __device__ constexpr explicit operator bool() const { return _is_valid; }

    __device__ constexpr T value_or(T v) const { return _is_valid ? _value : v; }
  };

  template <typename T>
  optional(T) -> optional<T>;

  template <typename T>
  struct span {
    T* _data = nullptr;

    size_t _size = 0;

    constexpr span() = default;

    __device__ constexpr span(T* data, size_t size) : _data{data}, _size{size} {}

    __device__ constexpr T* data() const { return _data; }

    __device__ constexpr size_t size() const { return _size; }

    __device__ constexpr bool empty() const { return _size == 0; }

    __device__ constexpr T& operator[](size_t pos) const { return _data[pos]; }

    __device__ constexpr T* begin() const { return _data; }

    __device__ constexpr T* end() const { return _data + _size; }

    __device__ constexpr T const* cbegin() const { return _data; }

    __device__ constexpr T const* cend() const { return _data + _size; }

    __device__ constexpr span<T const> as_const() const { return span<T const>{_data, _size}; }

    __device__ constexpr T& element(size_t i) const { return _data[i]; }
  };

  template <typename T>
  span(T*, size_t) -> span<T>;

  struct string_view {
    static constexpr size_type const UNKNOWN_STRING_LENGTH{-1};
    static constexpr size_type const npos{-1};

    char const* _data = "";

    size_type _bytes = 0;

    mutable size_type _length = UNKNOWN_STRING_LENGTH;

    constexpr string_view() = default;

    __device__ constexpr string_view(char const* data, size_type bytes) : _data{data}, _bytes{bytes}
    {
    }

    __device__ constexpr string_view(char const* data, size_type bytes, size_type length)
      : _data{data}, _bytes{bytes}, _length{length}
    {
    }

    __device__ constexpr size_type size_bytes() const { return _bytes; }

    __device__ constexpr auto* data() const { return _data; }

    __device__ constexpr auto* begin() const { return _data; }

    __device__ constexpr auto* end() const { return _data + _bytes; }

    __device__ constexpr auto const* cbegin() const { return _data; }

    __device__ constexpr auto const* cend() const { return _data + _bytes; }

    __device__ constexpr bool empty() const { return _bytes == 0; }

    __device__ constexpr size_type compare(string_view const& other) const
    {
      auto* s0 = _data;
      auto n0  = _bytes;
      auto* s1 = other._data;
      auto n1  = other._bytes;
      auto max = n0 < n1 ? n0 : n1;

      if (s0 == s1 && n0 == n1) return 0;

      size_type i = 0;

      while (i < max) {
        if (*s0 != *s1) return static_cast<int32_t>(*s0) - static_cast<int32_t>(*s1);
        s0++;
        s1++;
        i++;
      }

      if (i < n0) { return 1; }
      if (i < n1) { return -1; }

      return 0;
    }

    __device__ constexpr bool operator==(string_view const& rhs) const { return compare(rhs) == 0; }
    __device__ constexpr bool operator!=(string_view const& rhs) const { return compare(rhs) != 0; }
    __device__ constexpr bool operator<(string_view const& rhs) const { return compare(rhs) < 0; }
    __device__ constexpr bool operator<=(string_view const& rhs) const { return compare(rhs) <= 0; }
    __device__ constexpr bool operator>(string_view const& rhs) const { return compare(rhs) > 0; }
    __device__ constexpr bool operator>=(string_view const& rhs) const { return compare(rhs) >= 0; }
  };

  struct mutable_string_view {
    static constexpr size_type const UNKNOWN_STRING_LENGTH{-1};
    static constexpr size_type const npos{-1};

    char* _data = nullptr;

    size_type _bytes = 0;

    mutable size_type _length = UNKNOWN_STRING_LENGTH;

    constexpr mutable_string_view() = default;

    __device__ constexpr mutable_string_view(char* data, size_type bytes)
      : _data{data}, _bytes{bytes}
    {
    }

    __device__ constexpr size_type size_bytes() const { return _bytes; }

    __device__ constexpr auto* data() const { return _data; }

    __device__ constexpr auto* begin() const { return _data; }

    __device__ constexpr auto* end() const { return _data + _bytes; }

    __device__ constexpr auto const* cbegin() const { return _data; }

    __device__ constexpr auto const* cend() const { return _data + _bytes; }

    __device__ constexpr bool empty() const { return _bytes == 0; }

    __device__ explicit operator string_view() const { return string_view{_data, _bytes, _length}; }
  };

  // Aliases for codegen
  using b8     = bool;
  using i8     = int8_t;
  using i16    = int16_t;
  using i32    = int32_t;
  using i64    = int64_t;
  using u8     = uint8_t;
  using u16    = uint16_t;
  using u32    = uint32_t;
  using u64    = uint64_t;
  using f32    = float;
  using f64    = double;
  using dec32  = decimal32;
  using dec64  = decimal64;
  using dec128 = decimal128;
  using ts_D   = timestamp_D;
  using ts_h   = timestamp_h;
  using ts_m   = timestamp_m;
  using ts_s   = timestamp_s;
  using ts_ms  = timestamp_ms;
  using ts_us  = timestamp_us;
  using ts_ns  = timestamp_ns;
  using dur_D  = duration_D;
  using dur_h  = duration_h;
  using dur_m  = duration_m;
  using dur_s  = duration_s;
  using dur_ms = duration_ms;
  using dur_us = duration_us;
  using dur_ns = duration_ns;
  using str    = string_view;

  namespace operators {

  template <typename T>
  __device__ inline int abs(T* out, T const* a)
  {
    *out = (*a < 0) ? -*a : *a;
    return 0;
  }

  template <typename T>
  __device__ inline int abs(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      abs(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int add(T* out, T const* a, T const* b)
  {
    *out = (*a + *b);
    return 0;
  }

  template <typename T>
  __device__ inline int add(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      add(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arccos(T* out, T const* a);

  template <>
  __device__ inline int arccos<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::acosf(*a);
    return 0;
  }

  template <>
  __device__ inline int arccos<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::acos(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arccos(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arccos(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arccosh(T* out, T const* a);

  template <>
  __device__ inline int arccosh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::acoshf(*a);
    return 0;
  }

  template <>
  __device__ inline int arccosh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::acosh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arccosh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arccosh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arcsin(T* out, T const* a);

  template <>
  __device__ inline int arcsin<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::asinf(*a);
    return 0;
  }

  template <>
  __device__ inline int arcsin<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::asin(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arcsin(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arcsin(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arcsinh(T* out, T const* a);

  template <>
  __device__ inline int arcsinh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::asinhf(*a);
    return 0;
  }

  template <>
  __device__ inline int arcsinh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::asinh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arcsinh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arcsinh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arctan(T* out, T const* a);

  template <>
  __device__ inline int arctan<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::atanf(*a);
    return 0;
  }

  template <>
  __device__ inline int arctan<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::atan(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arctan(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arctan(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int arctanh(T* out, T const* a);

  template <>
  __device__ inline int arctanh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::atanhf(*a);
    return 0;
  }

  template <>
  __device__ inline int arctanh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::atanh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int arctanh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      arctanh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int bit_and(T* out, T const* a, T const* b)
  {
    *out = (*a & *b);
    return 0;
  }

  template <typename T>
  __device__ inline int bit_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      bit_and(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int bit_invert(T* out, T const* a)
  {
    *out = ~(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int bit_invert(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      bit_invert(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int bit_or(T* out, T const* a, T const* b)
  {
    *out = (*a | *b);
    return 0;
  }

  template <typename T>
  __device__ inline int bit_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      bit_or(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int bit_xor(T* out, T const* a, T const* b)
  {
    *out = (*a ^ *b);
    return 0;
  }

  template <typename T>
  __device__ inline int bit_xor(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      bit_xor(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_i32(int32_t* out, T const* a)
  {
    *out = static_cast<int32_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_i32(optional<int32_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      int32_t r;
      to_i32(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_i64(int64_t* out, T const* a)
  {
    *out = static_cast<int64_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_i64(optional<int64_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      int64_t r;
      to_i64(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_u32(uint32_t* out, T const* a)
  {
    *out = static_cast<uint32_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_u32(optional<uint32_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      uint32_t r;
      to_u32(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_u64(uint64_t* out, T const* a)
  {
    *out = static_cast<uint64_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_u64(optional<uint64_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      uint64_t r;
      to_u64(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_f32(float32_t* out, T const* a)
  {
    *out = static_cast<float32_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_f32(optional<float32_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      float32_t r;
      to_f32(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int to_f64(float64_t* out, T const* a)
  {
    *out = static_cast<float64_t>(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int to_f64(optional<float64_t>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      float64_t r;
      to_f64(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int cbrt(T* out, T const* a);

  template <>
  __device__ inline int cbrt<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::cbrtf(*a);
    return 0;
  }

  template <>
  __device__ inline int cbrt<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::cbrt(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int cbrt(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      cbrt(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int ceil(T* out, T const* a);

  template <>
  __device__ inline int ceil<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::ceilf(*a);
    return 0;
  }

  template <>
  __device__ inline int ceil<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::ceil(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int ceil(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      ceil(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int cos(T* out, T const* a);

  template <>
  __device__ inline int cos<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::cosf(*a);
    return 0;
  }

  template <>
  __device__ inline int cos<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::cos(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int cos(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      cos(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int cosh(T* out, T const* a);

  template <>
  __device__ inline int cosh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::coshf(*a);
    return 0;
  }

  template <>
  __device__ inline int cosh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::cosh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int cosh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      cosh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int div(T* out, T const* a, T const* b)
  {
    *out = (*a / *b);
    return 0;
  }

  template <typename T>
  __device__ inline int div(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      div(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int equal(bool* out, T const* a, T const* b)
  {
    *out = (*a == *b);
    return 0;
  }

  template <typename T>
  __device__ inline int equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      equal(&r, &a->value(), &b->value());
      *out = r;
    } else if (a->is_null() && b->is_null()) {
      *out = true;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int exp(T* out, T const* a);

  template <>
  __device__ inline int exp<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::expf(*a);
    return 0;
  }

  template <>
  __device__ inline int exp<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::exp(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int exp(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      exp(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int floor(T* out, T const* a);

  template <>
  __device__ inline int floor<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::floorf(*a);
    return 0;
  }

  template <>
  __device__ inline int floor<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::floor(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int floor(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      floor(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int greater(bool* out, T const* a, T const* b)
  {
    *out = (*a > *b);
    return 0;
  }

  template <typename T>
  __device__ inline int greater(optional<bool>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      greater(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int greater_equal(bool* out, T const* a, T const* b)
  {
    *out = (*a >= *b);
    return 0;
  }

  template <typename T>
  __device__ inline int greater_equal(optional<bool>* out,
                                      optional<T> const* a,
                                      optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      greater_equal(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int identity(T* out, T const* a)
  {
    *out = *a;
    return 0;
  }

  template <typename T>
  __device__ inline int identity(optional<T>* out, optional<T> const* a)
  {
    *out = *a;
    return 0;
  }

  template <typename T>
  __device__ inline int is_null(bool* out, T const* a)
  {
    *out = false;
    return 0;
  }

  template <typename T>
  __device__ inline int is_null(optional<bool>* out, optional<T> const* a)
  {
    *out = a->is_null();
    return 0;
  }

  template <typename T>
  __device__ inline int less(bool* out, T const* a, T const* b)
  {
    *out = (*a < *b);
    return 0;
  }

  template <typename T>
  __device__ inline int less(optional<bool>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      less(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int less_equal(bool* out, T const* a, T const* b)
  {
    *out = (*a <= *b);
    return 0;
  }

  template <typename T>
  __device__ inline int less_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      less_equal(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int log(T* out, T const* a);

  template <>
  __device__ inline int log<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::logf(*a);
    return 0;
  }

  template <>
  __device__ inline int log<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::log(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int log(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      log(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int logical_and(T* out, T const* a, T const* b)
  {
    *out = (*a && *b);
    return 0;
  }

  template <typename T>
  __device__ inline int logical_and(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      logical_and(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int logical_or(T* out, T const* a, T const* b)
  {
    *out = (*a || *b);
    return 0;
  }

  template <typename T>
  __device__ inline int logical_or(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      logical_or(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int mod(T* out, T const* a, T const* b)
  {
    *out = (*a % *b);
    return 0;
  }

  template <>
  __device__ inline int mod<float32_t>(float32_t* out, float32_t const* a, float32_t const* b)
  {
    *out = ::fmodf(*a, *b);
    return 0;
  }

  template <>
  __device__ inline int mod<float64_t>(float64_t* out, float64_t const* a, float64_t const* b)
  {
    *out = ::fmod(*a, *b);
    return 0;
  }

  template <typename T>
  __device__ inline int mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      mod(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int mul(T* out, T const* a, T const* b)
  {
    *out = (*a * *b);
    return 0;
  }

  template <typename T>
  __device__ inline int mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      mul(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int null_equal(bool* out, T const* a, T const* b)
  {
    *out = (*a == *b);
    return 0;
  }

  template <typename T>
  __device__ inline int null_equal(optional<bool>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      *out = (*(*a) == *(*b));
    } else if (a->is_null() && b->is_null()) {
      *out = true;
    } else {
      *out = false;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int null_logical_and(T* out, T const* a, T const* b)
  {
    *out = (*a && *b);
    return 0;
  }

  template <typename T>
  __device__ inline int null_logical_and(optional<T>* out,
                                         optional<T> const* a,
                                         optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      null_logical_and(&r, &a->value(), &b->value());
      *out = r;
    } else if (a->is_null() && b->is_null()) {
      *out = nullopt;
    } else {
      if (a->is_valid() ? *(*a) : *(*b)) {
        *out = nullopt;
      } else {
        *out = false;
      }
    }
    return 0;
  }

  template <typename T>
  __device__ inline int null_logical_or(T* out, T const* a, T const* b)
  {
    *out = (*a || *b);
    return 0;
  }

  template <typename T>
  __device__ inline int null_logical_or(optional<T>* out,
                                        optional<T> const* a,
                                        optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      bool r;
      null_logical_or(&r, &a->value(), &b->value());
      *out = r;
    } else if (a->is_null() && b->is_null()) {
      *out = nullopt;
    } else {
      if (a->is_valid() ? *(*a) : *(*b)) {
        *out = true;
      } else {
        *out = nullopt;
      }
    }
    return 0;
  }

  template <typename T>
  __device__ inline int pow(T* out, T const* a, T const* b);

  template <>
  __device__ inline int pow<float32_t>(float32_t* out, float32_t const* a, float32_t const* b)
  {
    *out = ::powf(*a, *b);
    return 0;
  }

  template <>
  __device__ inline int pow<float64_t>(float64_t* out, float64_t const* a, float64_t const* b)
  {
    *out = ::pow(*a, *b);
    return 0;
  }

  template <typename T>
  __device__ inline int pow(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      pow(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int pymod(T* out, T const* a, T const* b)
  {
    *out = (*a % *b + *b) % *b;
    return 0;
  }

  template <>
  __device__ inline int pymod<float32_t>(float32_t* out, float32_t const* a, float32_t const* b)
  {
    *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
    return 0;
  }

  template <>
  __device__ inline int pymod<float64_t>(float64_t* out, float64_t const* a, float64_t const* b)
  {
    *out = ::fmod(::fmod(*a, *b) + *b, *b);
    return 0;
  }

  template <typename T>
  __device__ inline int pymod(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      pymod(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int rint(T* out, T const* a);

  template <>
  __device__ inline int rint<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::rintf(*a);
    return 0;
  }

  template <>
  __device__ inline int rint<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::rint(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int rint(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      rint(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int sin(T* out, T const* a);

  template <>
  __device__ inline int sin<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::sinf(*a);
    return 0;
  }

  template <>
  __device__ inline int sin<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::sin(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int sin(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      sin(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int sinh(T* out, T const* a);

  template <>
  __device__ inline int sinh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::sinhf(*a);
    return 0;
  }

  template <>
  __device__ inline int sinh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::sinh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int sinh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      sinh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int sub(T* out, T const* a, T const* b)
  {
    *out = *a - *b;
    return 0;
  }

  template <typename T>
  __device__ inline int sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
  {
    if (a->is_valid() && b->is_valid()) {
      T r;
      sub(&r, &a->value(), &b->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int tanh(T* out, T const* a);

  template <>
  __device__ inline int tanh<float32_t>(float32_t* out, float32_t const* a)
  {
    *out = ::tanhf(*a);
    return 0;
  }

  template <>
  __device__ inline int tanh<float64_t>(float64_t* out, float64_t const* a)
  {
    *out = ::tanh(*a);
    return 0;
  }

  template <typename T>
  __device__ inline int tanh(optional<T>* out, optional<T> const* a)
  {
    if (a->is_valid()) {
      T r;
      tanh(&r, &a->value());
      *out = r;
    } else {
      *out = nullopt;
    }
    return 0;
  }

  template <typename T>
  __device__ inline int if_else(T* out,
                                bool const* condition,
                                T const* true_value,
                                T const* false_value)
  {
    *out = *condition ? *true_value : *false_value;
    return 0;
  }

  template <typename T>
  __device__ inline int if_else(optional<T>* out,
                                optional<bool> const* condition,
                                optional<T> const* true_value,
                                optional<T> const* false_value)
  {
    if (condition->is_valid() && true_value->is_valid() && false_value->is_valid()) {
      if_else<T>(&out->value(), &condition->value(), &true_value->value(), &false_value->value());
    } else {
      *out = nullopt;
    }
    return 0;
  }

  }  // namespace operators
  }  // namespace lite
}  // namespace cudf
