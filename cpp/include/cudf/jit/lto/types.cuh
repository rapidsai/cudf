/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#define CUDF_LTO_EXPORT
#define CUDF_LTO_ALIAS __attribute__((may_alias))

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

/**
 * @brief LTO-JIT functions and thunk types
 *
 * These are declarations for functions that will be used in LTO-JIT compiled code.
 * They are pre-compiled into a device library that is linked at JIT compile time.
 * This header should be minimal and only contain necessary types and function declarations as it
 * will be included and compiled at JIT compile time. Including other headers will lead to longer
 * JIT compile times which can be unbounded and cause slowdowns.
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
  type_id __id                = {};
  int32_t __fixed_point_scale = 0;
};

struct CUDF_LTO_ALIAS string_view {
 private:
  char const* __data         = nullptr;
  size_type __bytes          = 0;
  mutable size_type __length = 0;

 public:
  [[nodiscard]] __device__ inline size_type size_bytes() const;

  [[nodiscard]] __device__ inline size_type length() const;

  [[nodiscard]] __device__ inline char const* data() const;

  [[nodiscard]] __device__ inline bool empty() const;

  [[nodiscard]] __device__ inline char_utf8 operator[](size_type pos) const;

  [[nodiscard]] __device__ inline size_type byte_offset(size_type pos) const;

  [[nodiscard]] __device__ inline int compare(string_view const& str) const;

  [[nodiscard]] __device__ inline int compare(char const* str, size_type bytes) const;

  [[nodiscard]] __device__ inline bool operator==(string_view const& rhs) const;

  [[nodiscard]] __device__ inline bool operator!=(string_view const& rhs) const;

  [[nodiscard]] __device__ inline bool operator<(string_view const& rhs) const;

  [[nodiscard]] __device__ inline bool operator>(string_view const& rhs) const;

  [[nodiscard]] __device__ inline bool operator<=(string_view const& rhs) const;

  [[nodiscard]] __device__ inline bool operator>=(string_view const& rhs) const;

  [[nodiscard]] __device__ inline size_type find(string_view const& str,
                                                 size_type pos   = 0,
                                                 size_type count = -1) const;

  [[nodiscard]] __device__ inline size_type find(char const* str,
                                                 size_type bytes,
                                                 size_type pos   = 0,
                                                 size_type count = -1) const;

  [[nodiscard]] __device__ inline size_type find(char_utf8 character,
                                                 size_type pos   = 0,
                                                 size_type count = -1) const;

  [[nodiscard]] __device__ inline size_type rfind(string_view const& str,
                                                  size_type pos   = 0,
                                                  size_type count = -1) const;

  [[nodiscard]] __device__ inline size_type rfind(char const* str,
                                                  size_type bytes,
                                                  size_type pos   = 0,
                                                  size_type count = -1) const;

  [[nodiscard]] __device__ inline size_type rfind(char_utf8 character,
                                                  size_type pos   = 0,
                                                  size_type count = -1) const;

  [[nodiscard]] __device__ inline string_view substr(size_type start, size_type length) const;

  static inline size_type const npos{-1};
};

struct CUDF_LTO_ALIAS decimal32 {
 private:
  int32_t __value    = 0;
  scale_type __scale = scale_type{};
};

struct CUDF_LTO_ALIAS decimal64 {
 private:
  int64_t __value    = 0;
  scale_type __scale = scale_type{};
};

struct CUDF_LTO_ALIAS decimal128 {
 private:
  __int128_t __value = 0;
  scale_type __scale = scale_type{};
};

struct CUDF_LTO_ALIAS timestamp_D {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_h {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_m {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_s {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_ms {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_us {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS timestamp_ns {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_D {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_h {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_m {
 private:
  int32_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_s {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_ms {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_us {
 private:
  int64_t __rep = 0;
};

struct CUDF_LTO_ALIAS duration_ns {
 private:
  int64_t __rep = 0;
};

struct inplace_t {};

inline constexpr inplace_t inplace{};

// [ ] assumes T is trivially copyable
template <typename T>
struct CUDF_LTO_ALIAS optional {
 private:
  T __val;
  bool __engaged;

 public:
  __device__ constexpr optional() : __val{}, __engaged{false} {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : __val{static_cast<Args&&>(args)...}, __engaged{true}
  {
  }

  __device__ constexpr optional(T val) : __val{val}, __engaged{true} {}

  constexpr optional(optional const&) = default;

  constexpr optional(optional&&) = default;

  constexpr optional& operator=(optional const&) = default;

  constexpr optional& operator=(optional&&) = default;

  constexpr ~optional() = default;

  __device__ constexpr bool has_value() const { return __engaged; }

  __device__ constexpr void reset() { __engaged = false; }

  __device__ constexpr T const& get() const { return __val; }

  __device__ constexpr T& get() { return __val; }

  __device__ constexpr T const* operator->() const { return &__val; }

  __device__ constexpr T* operator->() { return &__val; }

  __device__ constexpr T const& operator*() const { return __val; }

  __device__ constexpr T& operator*() { return __val; }

  __device__ constexpr T const& value() const { return __val; }

  __device__ constexpr T& value() { return __val; }

  __device__ constexpr explicit operator bool() const { return __engaged; }

  __device__ constexpr T value_or(T __v) const { return __engaged ? __val : __v; }
};

template <typename T>
optional(T) -> optional<T>;

struct alignas(16) CUDF_LTO_ALIAS column_device_view_core {
 private:
  data_type __type                      = {};
  size_type __size                      = 0;
  void const* __data                    = nullptr;
  bitmask_type const* __null_mask       = nullptr;
  size_type __offset                    = 0;
  column_device_view_core* __d_children = nullptr;
  size_type __num_children              = 0;

 public:
  template <typename T>
  __device__ inline T const* head() const;

  __device__ inline size_type size() const;

  __device__ inline bool nullable() const;

  __device__ inline bitmask_type const* null_mask() const;

  __device__ inline size_type offset() const;

  __device__ inline bool is_valid(size_type idx) const;

  __device__ inline bool is_valid_nocheck(size_type idx) const;

  __device__ inline bool is_null(size_type idx) const;

  __device__ inline bool is_null_nocheck(size_type idx) const;

  __device__ inline bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ inline T element(size_type idx) const;

  __device__ inline size_type num_child_columns() const;
};

#define CUDF_LTO_DECL(Type) \
  extern template __device__ Type const* column_device_view_core::head<Type>() const;

CUDF_LTO_DECL(bool)
CUDF_LTO_DECL(int8_t)
CUDF_LTO_DECL(int16_t)
CUDF_LTO_DECL(int32_t)
CUDF_LTO_DECL(int64_t)
CUDF_LTO_DECL(uint8_t)
CUDF_LTO_DECL(uint16_t)
CUDF_LTO_DECL(uint32_t)
CUDF_LTO_DECL(uint64_t)
CUDF_LTO_DECL(float32_t)
CUDF_LTO_DECL(float64_t)
CUDF_LTO_DECL(timestamp_D)
CUDF_LTO_DECL(timestamp_h)
CUDF_LTO_DECL(timestamp_m)
CUDF_LTO_DECL(timestamp_s)
CUDF_LTO_DECL(timestamp_ms)
CUDF_LTO_DECL(timestamp_us)
CUDF_LTO_DECL(timestamp_ns)
CUDF_LTO_DECL(duration_D)
CUDF_LTO_DECL(duration_h)
CUDF_LTO_DECL(duration_m)
CUDF_LTO_DECL(duration_s)
CUDF_LTO_DECL(duration_ms)
CUDF_LTO_DECL(duration_us)
CUDF_LTO_DECL(duration_ns)

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(Type) \
  extern template __device__ Type column_device_view_core::element<Type>(size_type idx) const;

CUDF_LTO_DECL(bool)
CUDF_LTO_DECL(int8_t)
CUDF_LTO_DECL(int16_t)
CUDF_LTO_DECL(int32_t)
CUDF_LTO_DECL(int64_t)
CUDF_LTO_DECL(uint8_t)
CUDF_LTO_DECL(uint16_t)
CUDF_LTO_DECL(uint32_t)
CUDF_LTO_DECL(uint64_t)
CUDF_LTO_DECL(decimal32)
CUDF_LTO_DECL(decimal64)
CUDF_LTO_DECL(decimal128)
CUDF_LTO_DECL(float32_t)
CUDF_LTO_DECL(float64_t)
CUDF_LTO_DECL(string_view)
CUDF_LTO_DECL(timestamp_D)
CUDF_LTO_DECL(timestamp_h)
CUDF_LTO_DECL(timestamp_m)
CUDF_LTO_DECL(timestamp_s)
CUDF_LTO_DECL(timestamp_ms)
CUDF_LTO_DECL(timestamp_us)
CUDF_LTO_DECL(timestamp_ns)
CUDF_LTO_DECL(duration_D)
CUDF_LTO_DECL(duration_h)
CUDF_LTO_DECL(duration_m)
CUDF_LTO_DECL(duration_s)
CUDF_LTO_DECL(duration_ms)
CUDF_LTO_DECL(duration_us)
CUDF_LTO_DECL(duration_ns)

#undef CUDF_LTO_DECL

struct alignas(16) CUDF_LTO_ALIAS mutable_column_device_view_core {
 private:
  data_type __type                              = {};
  size_type __size                              = 0;
  void const* __data                            = nullptr;
  bitmask_type const* __null_mask               = nullptr;
  size_type __offset                            = 0;
  mutable_column_device_view_core* __d_children = nullptr;
  size_type __num_children                      = 0;

 public:
  template <typename T>
  __device__ inline T* head() const;

  __device__ inline size_type size() const;

  __device__ inline bool nullable() const;

  __device__ inline bitmask_type* null_mask() const;

  __device__ inline size_type offset() const;

  __device__ inline bool is_valid(size_type idx) const;

  __device__ inline bool is_valid_nocheck(size_type idx) const;

  __device__ inline bool is_null(size_type idx) const;

  __device__ inline bool is_null_nocheck(size_type idx) const;

  __device__ inline bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ inline T element(size_type idx) const;
};

#define CUDF_LTO_DECL(Type) \
  extern template __device__ Type* mutable_column_device_view_core::head<Type>() const;

CUDF_LTO_DECL(bool)
CUDF_LTO_DECL(int8_t)
CUDF_LTO_DECL(int16_t)
CUDF_LTO_DECL(int32_t)
CUDF_LTO_DECL(int64_t)
CUDF_LTO_DECL(uint8_t)
CUDF_LTO_DECL(uint16_t)
CUDF_LTO_DECL(uint32_t)
CUDF_LTO_DECL(uint64_t)
CUDF_LTO_DECL(float32_t)
CUDF_LTO_DECL(float64_t)
CUDF_LTO_DECL(timestamp_D)
CUDF_LTO_DECL(timestamp_h)
CUDF_LTO_DECL(timestamp_m)
CUDF_LTO_DECL(timestamp_s)
CUDF_LTO_DECL(timestamp_ms)
CUDF_LTO_DECL(timestamp_us)
CUDF_LTO_DECL(timestamp_ns)
CUDF_LTO_DECL(duration_D)
CUDF_LTO_DECL(duration_h)
CUDF_LTO_DECL(duration_m)
CUDF_LTO_DECL(duration_s)
CUDF_LTO_DECL(duration_ms)
CUDF_LTO_DECL(duration_us)
CUDF_LTO_DECL(duration_ns)

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(Type)                                                                     \
  extern template __device__ Type mutable_column_device_view_core::element<Type>(size_type idx) \
    const;

CUDF_LTO_DECL(bool)
CUDF_LTO_DECL(int8_t)
CUDF_LTO_DECL(int16_t)
CUDF_LTO_DECL(int32_t)
CUDF_LTO_DECL(int64_t)
CUDF_LTO_DECL(uint8_t)
CUDF_LTO_DECL(uint16_t)
CUDF_LTO_DECL(uint32_t)
CUDF_LTO_DECL(uint64_t)
CUDF_LTO_DECL(decimal32)
CUDF_LTO_DECL(decimal64)
CUDF_LTO_DECL(decimal128)
CUDF_LTO_DECL(float32_t)
CUDF_LTO_DECL(float64_t)
CUDF_LTO_DECL(string_view)
CUDF_LTO_DECL(timestamp_D)
CUDF_LTO_DECL(timestamp_h)
CUDF_LTO_DECL(timestamp_m)
CUDF_LTO_DECL(timestamp_s)
CUDF_LTO_DECL(timestamp_ms)
CUDF_LTO_DECL(timestamp_us)
CUDF_LTO_DECL(timestamp_ns)
CUDF_LTO_DECL(duration_D)
CUDF_LTO_DECL(duration_h)
CUDF_LTO_DECL(duration_m)
CUDF_LTO_DECL(duration_s)
CUDF_LTO_DECL(duration_ms)
CUDF_LTO_DECL(duration_us)
CUDF_LTO_DECL(duration_ns)

#undef CUDF_LTO_DECL

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
