/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

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

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
