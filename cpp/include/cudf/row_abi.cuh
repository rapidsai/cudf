/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/abi_types.cuh>

/**
 * @file
 * @brief Dependency-free ABI for separately compiled CUDA row operations.
 *
 * ABI version 1 uses CUDA natural alignment, 64-bit pointers, and the element
 * representations in abi_types.cuh. No implementation-library layout crosses
 * this boundary. Null payloads must not be inspected. Views borrow device memory
 * which the caller keeps alive on the execution stream.
 */
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CUDF_ROW_DEVICE __device__
#else
#define CUDF_ROW_DEVICE
#endif
namespace cudf::row_abi {
using namespace cudf::abi_types;
using string_view                           = cudf::abi_types::string_view;
inline constexpr u32 version                = 1;
using status                                = i32;
inline constexpr status success             = 0;
inline constexpr status arithmetic_overflow = 1;
inline constexpr status division_by_zero    = 2;

/** Transform and predicate entries own their semantic column access. */
extern "C" CUDF_ROW_DEVICE status transform_operator(void const* inputs,
                                                     void const* outputs,
                                                     void* user_data,
                                                     i32 row);
extern "C" CUDF_ROW_DEVICE status predicate_operator(void const* inputs,
                                                     bool8* outputs,
                                                     void* user_data,
                                                     i32 row);

/** Opaque handles; callers select the codec from the host schema. */
extern "C" CUDF_ROW_DEVICE void const* cudf_row_input_column(void const* columns, i32 index);
extern "C" CUDF_ROW_DEVICE void const* cudf_row_output_column(void const* columns, i32 index);
CUDF_ROW_DEVICE i32 load_list_size(void const* column, i32 row);
CUDF_ROW_DEVICE optional<i32> load_list_size_nullable(void const* column, i32 row);
CUDF_ROW_DEVICE i32 output_list_size(void const* column, i32 row);

/** Update a parent mask; prepared storage must have a mask when clearing validity. */
CUDF_ROW_DEVICE void set_validity(void const* columns, i32 col, i32 row, bool8 valid);
/** Update a list child's mask, preserving parent validity; null parents bypass child access. */
CUDF_ROW_DEVICE void set_validity(void const* columns, i32 col, i32 row, i32 element, bool8 valid);

/**
 * Payload access never dispatches on a device-side type ID. Nullable accessors
 * load validity together with optional payloads; plain accessors omit mask access. Indices
 * and elements must be in range. Tags select physical dictionary indices and string offsets.
 * Payload setters do not change validity. Validity setters update masks cooperatively within each
 * warp. Dictionary setters select an existing key index and leave the shared keys unchanged. Output
 * rows/elements and key indices must be in range and representable. Prepared string byte counts and
 * decimal scales must match the payload. A mask is required to write nulls.
 */
#define CUDF_ROW_DECLARE_DICT(T, I, ...)                                                      \
  CUDF_ROW_DEVICE T load_dict_##T(void const*, i32, i32, index_type_tag<I> __VA_ARGS__);      \
  CUDF_ROW_DEVICE T load_dict_list_##T(void const*, i32, i32, index_type_tag<I> __VA_ARGS__); \
  CUDF_ROW_DEVICE status set_dict_##T(                                                        \
    void const*, i32, i32, dictionary_index, index_type_tag<I> __VA_ARGS__);                  \
  CUDF_ROW_DEVICE status set_dict_list_##T(                                                   \
    void const*, i32, i32, dictionary_index, index_type_tag<I> __VA_ARGS__);                  \
  CUDF_ROW_DEVICE optional<T> load_dict_##T##_nullable(                                       \
    void const*, i32, i32, index_type_tag<I> __VA_ARGS__);                                    \
  CUDF_ROW_DEVICE optional<T> load_dict_list_##T##_nullable(                                  \
    void const*, i32, i32, index_type_tag<I> __VA_ARGS__);
#define CUDF_ROW_DECLARE_INDICES(T, ...)     \
  CUDF_ROW_DECLARE_DICT(T, i8, __VA_ARGS__)  \
  CUDF_ROW_DECLARE_DICT(T, i16, __VA_ARGS__) \
  CUDF_ROW_DECLARE_DICT(T, i32, __VA_ARGS__) \
  CUDF_ROW_DECLARE_DICT(T, i64, __VA_ARGS__) \
  CUDF_ROW_DECLARE_DICT(T, u8, __VA_ARGS__)  \
  CUDF_ROW_DECLARE_DICT(T, u16, __VA_ARGS__) \
  CUDF_ROW_DECLARE_DICT(T, u32, __VA_ARGS__) \
  CUDF_ROW_DECLARE_DICT(T, u64, __VA_ARGS__)
#define CUDF_ROW_DECLARE_ACCESS(T, ...)                                                    \
  CUDF_ROW_DEVICE T load_##T(void const*, i32, i32 __VA_ARGS__);                           \
  CUDF_ROW_DEVICE T load_list_##T(void const*, i32, i32 __VA_ARGS__);                      \
  CUDF_ROW_DEVICE status set_##T(void const*, i32, i32, T __VA_ARGS__);                    \
  CUDF_ROW_DEVICE status set_list_##T(void const*, i32, i32, T __VA_ARGS__);               \
  CUDF_ROW_DEVICE optional<T> load_##T##_nullable(void const*, i32, i32 __VA_ARGS__);      \
  CUDF_ROW_DEVICE optional<T> load_list_##T##_nullable(void const*, i32, i32 __VA_ARGS__); \
  CUDF_ROW_DECLARE_INDICES(T, __VA_ARGS__)
CUDF_ROW_DECLARE_ACCESS(bool8)
CUDF_ROW_DECLARE_ACCESS(i8)
CUDF_ROW_DECLARE_ACCESS(i16)
CUDF_ROW_DECLARE_ACCESS(i32)
CUDF_ROW_DECLARE_ACCESS(i64)
CUDF_ROW_DECLARE_ACCESS(u8)
CUDF_ROW_DECLARE_ACCESS(u16)
CUDF_ROW_DECLARE_ACCESS(u32)
CUDF_ROW_DECLARE_ACCESS(u64)
CUDF_ROW_DECLARE_ACCESS(f32)
CUDF_ROW_DECLARE_ACCESS(f64)
CUDF_ROW_DECLARE_ACCESS(decimal32)
CUDF_ROW_DECLARE_ACCESS(decimal64)
CUDF_ROW_DECLARE_ACCESS(decimal128)
CUDF_ROW_DECLARE_ACCESS(timestamp_days)
CUDF_ROW_DECLARE_ACCESS(timestamp_seconds)
CUDF_ROW_DECLARE_ACCESS(timestamp_milliseconds)
CUDF_ROW_DECLARE_ACCESS(timestamp_microseconds)
CUDF_ROW_DECLARE_ACCESS(timestamp_nanoseconds)
CUDF_ROW_DECLARE_ACCESS(duration_days)
CUDF_ROW_DECLARE_ACCESS(duration_seconds)
CUDF_ROW_DECLARE_ACCESS(duration_milliseconds)
CUDF_ROW_DECLARE_ACCESS(duration_microseconds)
CUDF_ROW_DECLARE_ACCESS(duration_nanoseconds)
#define CUDF_ROW_OFFSET_TAG(O) , type_tag<O>
CUDF_ROW_DECLARE_ACCESS(string_view, CUDF_ROW_OFFSET_TAG(i32))
CUDF_ROW_DECLARE_ACCESS(string_view, CUDF_ROW_OFFSET_TAG(i64))
#undef CUDF_ROW_OFFSET_TAG
#undef CUDF_ROW_DECLARE_ACCESS
#undef CUDF_ROW_DECLARE_INDICES
#undef CUDF_ROW_DECLARE_DICT

#define CUDF_ROW_DECLARE_DICT_SIZE(I)                                           \
  CUDF_ROW_DEVICE i32 load_dict_list_size(void const*, i32, index_type_tag<I>); \
  CUDF_ROW_DEVICE optional<i32> load_dict_list_size_nullable(void const*, i32, index_type_tag<I>);
CUDF_ROW_DECLARE_DICT_SIZE(i8)
CUDF_ROW_DECLARE_DICT_SIZE(i16)
CUDF_ROW_DECLARE_DICT_SIZE(i32)
CUDF_ROW_DECLARE_DICT_SIZE(i64)
CUDF_ROW_DECLARE_DICT_SIZE(u8)
CUDF_ROW_DECLARE_DICT_SIZE(u16)
CUDF_ROW_DECLARE_DICT_SIZE(u32)
CUDF_ROW_DECLARE_DICT_SIZE(u64)
#undef CUDF_ROW_DECLARE_DICT_SIZE

/** Direct access to caller-prepared string character storage. */
#define CUDF_ROW_DECLARE_STRING_STORAGE(O)                          \
  CUDF_ROW_DEVICE char* string_data(void const*, i32, type_tag<O>); \
  CUDF_ROW_DEVICE i64 string_size(void const*, i32, type_tag<O>);
CUDF_ROW_DECLARE_STRING_STORAGE(i32)
CUDF_ROW_DECLARE_STRING_STORAGE(i64)
#undef CUDF_ROW_DECLARE_STRING_STORAGE

/** Store a borrowed view into an allocating transform's temporary view buffer. */
CUDF_ROW_DEVICE status set_string_view(void const*, i32, i32, string_view);
}  // namespace cudf::row_abi
#undef CUDF_ROW_DEVICE
