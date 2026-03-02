/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
namespace jit {
namespace cudf {
namespace functions {

__device__ inline void null_logical_and(bool* out, bool const& a, bool const& b) { *out = a && b; }

__device__ inline void null_logical_and(optional<bool>* out,
                                        optional<bool> const& a,
                                        optional<bool> const& b)
{
  if (a.has_value() && b.has_value()) {
    *out = (*a && *b);
  } else if (a.has_null() && b.has_null()) {
    *out = nullopt;
  } else {
    bool valid = a.has_value() ? *a : *b;
    if (valid) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
}

}  // namespace functions
}  // namespace cudf
}  // namespace jit
