/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf::hashing::detail::jit_lto {

struct tag_murmur_entry {};

struct tag_i8 {};
struct tag_i16 {};
struct tag_i32 {};
struct tag_i64 {};
struct tag_u8 {};
struct tag_u16 {};
struct tag_u32 {};
struct tag_u64 {};
struct tag_f32 {};
struct tag_f64 {};
struct tag_b8 {};
struct tag_ts_day {};
struct tag_ts_s {};
struct tag_ts_ms {};
struct tag_ts_us {};
struct tag_ts_ns {};
struct tag_du_day {};
struct tag_du_s {};
struct tag_du_ms {};
struct tag_du_us {};
struct tag_du_ns {};
struct tag_dict {};
struct tag_str {};
struct tag_list {};
struct tag_dec32 {};
struct tag_dec64 {};
struct tag_dec128 {};
struct tag_struct {};

template <typename StorageTag>
struct fragment_tag_murmur_hasher {};

/// Strong `murmur_jit_hasher<T>` with a no-op body; used for storage types not present in the
/// table so nvJitLink still sees exactly one definition per `T` (weak overrides are not supported).
template <typename StorageTag>
struct fragment_tag_murmur_hasher_noop {};

struct fragment_tag_murmur_entry {};

}  // namespace cudf::hashing::detail::jit_lto
