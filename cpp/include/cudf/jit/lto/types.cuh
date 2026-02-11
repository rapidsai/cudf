/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#define CUDF_LTO_EXPORT __attribute__((visibility("default")))
#define CUDF_LTO_ALIAS  __attribute__((may_alias))

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

struct CUDF_LTO_ALIAS string_view {
 private:
  char const* _data         = nullptr;
  size_type _bytes          = 0;
  mutable size_type _length = 0;

 public:
  [[nodiscard]] __device__ size_type size_bytes() const;

  [[nodiscard]] __device__ size_type length() const;

  [[nodiscard]] __device__ char const* data() const;

  [[nodiscard]] __device__ bool empty() const;

  [[nodiscard]] __device__ char_utf8 operator[](size_type pos) const;

  [[nodiscard]] __device__ size_type byte_offset(size_type pos) const;

  [[nodiscard]] __device__ int compare(string_view const& str) const;

  [[nodiscard]] __device__ int compare(char const* str, size_type bytes) const;

  [[nodiscard]] __device__ bool operator==(string_view const& rhs) const;

  [[nodiscard]] __device__ bool operator!=(string_view const& rhs) const;

  [[nodiscard]] __device__ bool operator<(string_view const& rhs) const;

  [[nodiscard]] __device__ bool operator>(string_view const& rhs) const;

  [[nodiscard]] __device__ bool operator<=(string_view const& rhs) const;

  [[nodiscard]] __device__ bool operator>=(string_view const& rhs) const;

  [[nodiscard]] __device__ size_type find(string_view const& str,
                                          size_type pos   = 0,
                                          size_type count = -1) const;

  [[nodiscard]] __device__ size_type find(char const* str,
                                          size_type bytes,
                                          size_type pos   = 0,
                                          size_type count = -1) const;

  [[nodiscard]] __device__ size_type find(char_utf8 character,
                                          size_type pos   = 0,
                                          size_type count = -1) const;

  [[nodiscard]] __device__ size_type rfind(string_view const& str,
                                           size_type pos   = 0,
                                           size_type count = -1) const;

  [[nodiscard]] __device__ size_type rfind(char const* str,
                                           size_type bytes,
                                           size_type pos   = 0,
                                           size_type count = -1) const;

  [[nodiscard]] __device__ size_type rfind(char_utf8 character,
                                           size_type pos   = 0,
                                           size_type count = -1) const;

  [[nodiscard]] __device__ string_view substr(size_type start, size_type length) const;

  static inline size_type const npos{-1};
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

struct inplace_t {};

inline constexpr inplace_t inplace{};

// TODO: assumes T is trivially copyable
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

template <typename T>
struct [[nodiscard]] device_span {
 private:
  T* _data     = nullptr;
  size_t _size = 0;
};

__device__ constexpr bool bit_is_set(bitmask_type const* bitmask, size_t bit_index)
{
  constexpr auto bits_per_word = sizeof(bitmask_type) * 8;
  return bitmask[bit_index / bits_per_word] & (bitmask_type{1} << (bit_index % bits_per_word));
}

template <typename T>
struct [[nodiscard]] device_optional_span {
 private:
  T* _data                 = nullptr;
  size_t _size             = 0;
  bitmask_type* _null_mask = nullptr;

 public:
  __device__ T* data() const { return _data; }

  __device__ size_t size() const { return _size; }

  __device__ bool empty() const { return _size == 0; }

  __device__ T& operator[](size_t pos) const { return _data[pos]; }

  __device__ T* begin() const { return _data; }

  __device__ T* end() const { return _data + _size; }

  __device__ device_optional_span<T const> as_const() const
  {
    return device_optional_span<T const>{_data, _size, _null_mask};
  }

  __device__ bool nullable() const { return _null_mask != nullptr; }

  [[nodiscard]] __device__ bool is_valid_nocheck(size_t element_index) const
  {
    return bit_is_set(_null_mask, element_index);
  }

  __device__ bool is_valid(size_t element_index) const
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  __device__ bool is_null(size_t element_index) const { return !is_valid(element_index); }

  __device__ T& element(size_t idx) const { return _data[idx]; }

  __device__ optional<T> nullable_element(size_t idx) const;

  __device__ void assign(size_t idx, T value) const { _data[idx] = value; }
};

#define FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE \
  DO_IT(bool)                             \
  DO_IT(int8_t)                           \
  DO_IT(int16_t)                          \
  DO_IT(int32_t)                          \
  DO_IT(int64_t)                          \
  DO_IT(uint8_t)                          \
  DO_IT(uint16_t)                         \
  DO_IT(uint32_t)                         \
  DO_IT(uint64_t)                         \
  DO_IT(float32_t)                        \
  DO_IT(float64_t)                        \
  DO_IT(timestamp_D)                      \
  DO_IT(timestamp_h)                      \
  DO_IT(timestamp_m)                      \
  DO_IT(timestamp_s)                      \
  DO_IT(timestamp_ms)                     \
  DO_IT(timestamp_us)                     \
  DO_IT(timestamp_ns)                     \
  DO_IT(duration_D)                       \
  DO_IT(duration_h)                       \
  DO_IT(duration_m)                       \
  DO_IT(duration_s)                       \
  DO_IT(duration_ms)                      \
  DO_IT(duration_us)                      \
  DO_IT(duration_ns)

#define FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE \
  DO_IT(bool)                                \
  DO_IT(int8_t)                              \
  DO_IT(int16_t)                             \
  DO_IT(int32_t)                             \
  DO_IT(int64_t)                             \
  DO_IT(uint8_t)                             \
  DO_IT(uint16_t)                            \
  DO_IT(uint32_t)                            \
  DO_IT(uint64_t)                            \
  DO_IT(decimal32)                           \
  DO_IT(decimal64)                           \
  DO_IT(decimal128)                          \
  DO_IT(float32_t)                           \
  DO_IT(float64_t)                           \
  DO_IT(string_view)                         \
  DO_IT(timestamp_D)                         \
  DO_IT(timestamp_h)                         \
  DO_IT(timestamp_m)                         \
  DO_IT(timestamp_s)                         \
  DO_IT(timestamp_ms)                        \
  DO_IT(timestamp_us)                        \
  DO_IT(timestamp_ns)                        \
  DO_IT(duration_D)                          \
  DO_IT(duration_h)                          \
  DO_IT(duration_m)                          \
  DO_IT(duration_s)                          \
  DO_IT(duration_ms)                         \
  DO_IT(duration_us)                         \
  DO_IT(duration_ns)

#define FOREACH_CUDF_LTO_COLUMN_ASSIGN_TYPE \
  DO_IT(bool)                               \
  DO_IT(int8_t)                             \
  DO_IT(int16_t)                            \
  DO_IT(int32_t)                            \
  DO_IT(int64_t)                            \
  DO_IT(uint8_t)                            \
  DO_IT(uint16_t)                           \
  DO_IT(uint32_t)                           \
  DO_IT(uint64_t)                           \
  DO_IT(decimal32)                          \
  DO_IT(decimal64)                          \
  DO_IT(decimal128)                         \
  DO_IT(float32_t)                          \
  DO_IT(float64_t)                          \
  DO_IT(timestamp_D)                        \
  DO_IT(timestamp_h)                        \
  DO_IT(timestamp_m)                        \
  DO_IT(timestamp_s)                        \
  DO_IT(timestamp_ms)                       \
  DO_IT(timestamp_us)                       \
  DO_IT(timestamp_ns)                       \
  DO_IT(duration_D)                         \
  DO_IT(duration_h)                         \
  DO_IT(duration_m)                         \
  DO_IT(duration_s)                         \
  DO_IT(duration_ms)                        \
  DO_IT(duration_us)                        \
  DO_IT(duration_ns)

struct alignas(16) CUDF_LTO_ALIAS column_device_view_core {
 private:
  data_type _type                      = {};
  size_type _size                      = 0;
  void const* _data                    = nullptr;
  bitmask_type const* _null_mask       = nullptr;
  size_type _offset                    = 0;
  column_device_view_core* _d_children = nullptr;
  size_type _num_children              = 0;

 public:
  template <typename T>
  __device__ T const* head() const;

  __device__ size_type size() const;

  __device__ bool nullable() const;

  __device__ bitmask_type const* null_mask() const;

  __device__ size_type offset() const;

  __device__ bool is_valid(size_type idx) const;

  __device__ bool is_valid_nocheck(size_type idx) const;

  __device__ bool is_null(size_type idx) const;

  __device__ bool is_null_nocheck(size_type idx) const;

  __device__ bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ T element(size_type idx) const;

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const;

  __device__ size_type num_child_columns() const;
};

struct alignas(16) CUDF_LTO_ALIAS mutable_column_device_view_core {
 private:
  data_type _type                              = {};
  size_type _size                              = 0;
  void const* _data                            = nullptr;
  bitmask_type const* _null_mask               = nullptr;
  size_type _offset                            = 0;
  mutable_column_device_view_core* _d_children = nullptr;
  size_type _num_children                      = 0;

 public:
  template <typename T>
  __device__ T* head() const;

  __device__ size_type size() const;

  __device__ bool nullable() const;

  __device__ bitmask_type* null_mask() const;

  __device__ size_type offset() const;

  __device__ bool is_valid(size_type idx) const;

  __device__ bool is_valid_nocheck(size_type idx) const;

  __device__ bool is_null(size_type idx) const;

  __device__ bool is_null_nocheck(size_type idx) const;

  __device__ bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ T element(size_type idx) const;

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const;

  template <typename T>
  __device__ void assign(size_type idx, T value) const;
};

#define DO_IT(T) extern template __device__ T const* column_device_view_core::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) \
  extern template __device__ T column_device_view_core::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                       \
  extern template __device__ optional<T> column_device_view_core::nullable_element<T>( \
    size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) extern template __device__ T* mutable_column_device_view_core::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) \
  extern template __device__ T mutable_column_device_view_core::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                               \
  extern template __device__ optional<T> mutable_column_device_view_core::nullable_element<T>( \
    size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                            \
  extern template __device__ void mutable_column_device_view_core::assign<T>(size_type idx, \
                                                                             T value) const;

FOREACH_CUDF_LTO_COLUMN_ASSIGN_TYPE

#undef DO_IT

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
