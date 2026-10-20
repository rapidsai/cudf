/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "codecs.cuh"

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/lists/lists_column_view.hpp>

#include <cuda/atomic>

#include <jit/column_device_view_wrappers.cuh>

namespace cudf::row_abi {
namespace {
using input_column  = column_device_view_core;
using output_column = mutable_column_device_view_core;
template <typename Column>
__device__ i32 list_begin(Column const& column, i32 row)
{
  return column.child(offsets_column_index).template element<int32_t>(row + column.offset());
}
template <typename I, typename Column>
__device__ i32 dict_row(Column const& column, i32 row)
{
  using P = typename detail::codec<I>::physical_type;
  return static_cast<i32>(
    column.child(dictionary_indices_column_index).template element<P>(row + column.offset()));
}
template <typename T, typename O = i32>
__device__ T payload(input_column const& column, i32 row)
{
  if constexpr (cuda::std::is_same_v<T, string_view>) {
    using P            = typename detail::codec<O>::physical_type;
    auto const offsets = column.child(offsets_column_index);
    auto const begin   = offsets.template element<P>(row + column.offset());
    auto const end     = offsets.template element<P>(row + column.offset() + 1);
    auto const chars   = column.head<char>();
    return {chars == nullptr ? nullptr : chars + begin, static_cast<i32>(end - begin), -1};
  } else {
    using P = typename detail::codec<T>::physical_type;
    return detail::codec<T>::encode(column.template element<P>(row));
  }
}
template <typename O>
__device__ i64 bytes(output_column const& column, i32 row)
{
  using P            = typename detail::codec<O>::physical_type;
  auto const offsets = column.child(offsets_column_index);
  return offsets.template element<P>(row + column.offset() + 1) -
         offsets.template element<P>(row + column.offset());
}
template <typename O>
__device__ char* chars(output_column const& column, i32 row)
{
  using P          = typename detail::codec<O>::physical_type;
  auto const begin = column.child(offsets_column_index).template element<P>(row + column.offset());
  auto const data  = column.head<char>();
  return data == nullptr ? nullptr : data + begin;
}
template <typename T, typename O = i32>
__device__ status assign(output_column const& column, i32 row, T value)
{
  if constexpr (cuda::std::is_same_v<T, string_view>) {
    auto const data = chars<O>(column, row);
    for (i32 i = 0; i < value.bytes; ++i) {
      data[i] = value.data[i];
    }
  } else {
    column.assign(row, detail::codec<T>::decode(value));
  }
  return success;
}
template <typename I>
__device__ status assign_index(output_column const& column, i32 row, dictionary_index value)
{
  using P = typename detail::codec<I>::physical_type;
  column.child(dictionary_indices_column_index)
    .template assign<P>(row + column.offset(), static_cast<P>(value));
  return success;
}

template <typename T, typename O>
__device__ optional<T> nullable_payload(input_column const& column, i32 row)
{
  return column.is_null(row) ? optional<T>{} : optional<T>{payload<T, O>(column, row), 1};
}
template <typename T, typename O>
__device__ optional<T> nullable_list_payload(input_column const& column, i32 row, i32 element)
{
  if (column.is_null(row)) { return {}; }
  return nullable_payload<T, O>(column.child(lists_column_view::child_column_index),
                                list_begin(column, row) + element);
}
}  // namespace
#define CUDF_ROW_EXPORT __device__ __inline_hint__
#define CUDF_ROW_INPUT_COLUMN()                                                                \
  extern "C" CUDF_ROW_EXPORT void const* cudf_row_input_column(void const* columns, i32 index) \
  {                                                                                            \
    return static_cast<input_column const*>(columns) + index;                                  \
  }
#define CUDF_ROW_OUTPUT_COLUMN()                                                                \
  extern "C" CUDF_ROW_EXPORT void const* cudf_row_output_column(void const* columns, i32 index) \
  {                                                                                             \
    return static_cast<output_column const*>(columns) + index;                                  \
  }
#define CUDF_ROW_LOAD_LIST_SIZE()                                 \
  CUDF_ROW_EXPORT i32 load_list_size(void const* column, i32 row) \
  {                                                               \
    auto const& view = *static_cast<input_column const*>(column); \
    return list_begin(view, row + 1) - list_begin(view, row);     \
  }
#define CUDF_ROW_OUTPUT_LIST_SIZE()                                 \
  CUDF_ROW_EXPORT i32 output_list_size(void const* column, i32 row) \
  {                                                                 \
    auto const& view = *static_cast<output_column const*>(column);  \
    return list_begin(view, row + 1) - list_begin(view, row);       \
  }
#define CUDF_ROW_SET_STRING_VIEW()                                                                 \
  CUDF_ROW_EXPORT status set_string_view(void const* columns, i32 col, i32 row, string_view value) \
  {                                                                                                \
    jit::mutable_vector_device_view{static_cast<output_column const*>(columns)[col]}.assign(       \
      row, detail::codec<string_view>::decode(value));                                             \
    return success;                                                                                \
  }
#define CUDF_ROW_SET_VALIDITY()                                                         \
  CUDF_ROW_EXPORT void set_validity(void const* columns, i32 col, i32 row, bool8 valid) \
  {                                                                                     \
    jit::set_validity_warp_cooperative(                                                 \
      static_cast<output_column const*>(columns)[col], row, valid);                     \
  }
#define CUDF_ROW_SET_LIST_VALIDITY()                                                              \
  CUDF_ROW_EXPORT void set_validity(                                                              \
    void const* columns, i32 col, i32 row, i32 element, bool8 valid)                              \
  {                                                                                               \
    auto const& view = static_cast<output_column const*>(columns)[col];                           \
    if (view.is_null(row)) { return; }                                                            \
    jit::set_validity_warp_cooperative(                                                           \
      view.child(lists_column_view::child_column_index), list_begin(view, row) + element, valid); \
  }
#define CUDF_ROW_LOAD_LIST_SIZE_NULLABLE()                                           \
  CUDF_ROW_EXPORT optional<i32> load_list_size_nullable(void const* column, i32 row) \
  {                                                                                  \
    auto const& view = *static_cast<input_column const*>(column);                    \
    return view.is_null(row)                                                         \
             ? optional<i32>{}                                                       \
             : optional<i32>{list_begin(view, row + 1) - list_begin(view, row), 1};  \
  }
#define CUDF_ROW_LOAD_DICT_LIST_SIZE(I)                                                   \
  CUDF_ROW_EXPORT i32 load_dict_list_size(void const* column, i32 row, index_type_tag<I>) \
  {                                                                                       \
    auto const& view   = *static_cast<input_column const*>(column);                       \
    auto const keys    = view.child(dictionary_keys_column_index);                        \
    auto const key_row = dict_row<I>(view, row);                                          \
    return list_begin(keys, key_row + 1) - list_begin(keys, key_row);                     \
  }
#define CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(I)                                            \
  CUDF_ROW_EXPORT optional<i32> load_dict_list_size_nullable(                               \
    void const* column, i32 row, index_type_tag<I>)                                         \
  {                                                                                         \
    auto const& view = *static_cast<input_column const*>(column);                           \
    if (view.is_null(row)) { return {}; }                                                   \
    auto const keys    = view.child(dictionary_keys_column_index);                          \
    auto const key_row = dict_row<I>(view, row);                                            \
    return keys.is_null(key_row)                                                            \
             ? optional<i32>{}                                                              \
             : optional<i32>{list_begin(keys, key_row + 1) - list_begin(keys, key_row), 1}; \
  }
#define CUDF_ROW_LOAD(T, O, ...)                                                \
  CUDF_ROW_EXPORT T load_##T(void const* columns, i32 col, i32 row __VA_ARGS__) \
  {                                                                             \
    return payload<T, O>(static_cast<input_column const*>(columns)[col], row);  \
  }
#define CUDF_ROW_LOAD_LIST(T, O, ...)                                                   \
  CUDF_ROW_EXPORT T load_list_##T(void const* column, i32 row, i32 element __VA_ARGS__) \
  {                                                                                     \
    auto const& view = *static_cast<input_column const*>(column);                       \
    return payload<T, O>(view.child(lists_column_view::child_column_index),             \
                         list_begin(view, row) + element);                              \
  }
#define CUDF_ROW_SET(T, O, ...)                                                              \
  CUDF_ROW_EXPORT status set_##T(void const* columns, i32 col, i32 row, T value __VA_ARGS__) \
  {                                                                                          \
    return assign<T, O>(static_cast<output_column const*>(columns)[col], row, value);        \
  }
#define CUDF_ROW_SET_LIST(T, O, ...)                                                              \
  CUDF_ROW_EXPORT status set_list_##T(                                                            \
    void const* column, i32 row, i32 element, T value __VA_ARGS__)                                \
  {                                                                                               \
    auto const& view = *static_cast<output_column const*>(column);                                \
    return assign<T, O>(                                                                          \
      view.child(lists_column_view::child_column_index), list_begin(view, row) + element, value); \
  }
#define CUDF_ROW_LOAD_NULLABLE(T, O, ...)                              \
  CUDF_ROW_EXPORT optional<T> load_##T##_nullable(                     \
    void const* columns, i32 col, i32 row __VA_ARGS__)                 \
  {                                                                    \
    auto const& view = static_cast<input_column const*>(columns)[col]; \
    return nullable_payload<T, O>(view, row);                          \
  }
#define CUDF_ROW_LOAD_LIST_NULLABLE(T, O, ...)                                                   \
  CUDF_ROW_EXPORT optional<T> load_list_##T##_nullable(                                          \
    void const* column, i32 row, i32 element __VA_ARGS__)                                        \
  {                                                                                              \
    return nullable_list_payload<T, O>(*static_cast<input_column const*>(column), row, element); \
  }
#define CUDF_ROW_PLAIN_ACCESS(T, O, ...)    \
  CUDF_ROW_LOAD(T, O, __VA_ARGS__)          \
  CUDF_ROW_LOAD_LIST(T, O, __VA_ARGS__)     \
  CUDF_ROW_SET(T, O, __VA_ARGS__)           \
  CUDF_ROW_SET_LIST(T, O, __VA_ARGS__)      \
  CUDF_ROW_LOAD_NULLABLE(T, O, __VA_ARGS__) \
  CUDF_ROW_LOAD_LIST_NULLABLE(T, O, __VA_ARGS__)
#define CUDF_ROW_LOAD_DICT(T, I, O, ...)                                                    \
  CUDF_ROW_EXPORT T load_dict_##T(                                                          \
    void const* columns, i32 col, i32 row, index_type_tag<I> __VA_ARGS__)                   \
  {                                                                                         \
    auto const& view = static_cast<input_column const*>(columns)[col];                      \
    return payload<T, O>(view.child(dictionary_keys_column_index), dict_row<I>(view, row)); \
  }
#define CUDF_ROW_LOAD_DICT_LIST(T, I, O, ...)                                 \
  CUDF_ROW_EXPORT T load_dict_list_##T(                                       \
    void const* column, i32 row, i32 element, index_type_tag<I> __VA_ARGS__)  \
  {                                                                           \
    auto const& view = *static_cast<input_column const*>(column);             \
    auto const keys  = view.child(dictionary_keys_column_index);              \
    return payload<T, O>(keys.child(lists_column_view::child_column_index),   \
                         list_begin(keys, dict_row<I>(view, row)) + element); \
  }
#define CUDF_ROW_SET_DICT(T, I, O, ...)                                                           \
  CUDF_ROW_EXPORT status set_dict_##T(                                                            \
    void const* columns, i32 col, i32 row, dictionary_index value, index_type_tag<I> __VA_ARGS__) \
  {                                                                                               \
    return assign_index<I>(static_cast<output_column const*>(columns)[col], row, value);          \
  }
#define CUDF_ROW_SET_DICT_LIST(T, I, O, ...)                                                      \
  CUDF_ROW_EXPORT status set_dict_list_##T(                                                       \
    void const* columns, i32 col, i32 row, dictionary_index value, index_type_tag<I> __VA_ARGS__) \
  {                                                                                               \
    return assign_index<I>(static_cast<output_column const*>(columns)[col], row, value);          \
  }
#define CUDF_ROW_LOAD_DICT_NULLABLE(T, I, O, ...)                           \
  CUDF_ROW_EXPORT optional<T> load_dict_##T##_nullable(                     \
    void const* columns, i32 col, i32 row, index_type_tag<I> __VA_ARGS__)   \
  {                                                                         \
    auto const& view = static_cast<input_column const*>(columns)[col];      \
    if (view.is_null(row)) { return {}; }                                   \
    return nullable_payload<T, O>(view.child(dictionary_keys_column_index), \
                                  dict_row<I>(view, row));                  \
  }
#define CUDF_ROW_LOAD_DICT_LIST_NULLABLE(T, I, O, ...)                            \
  CUDF_ROW_EXPORT optional<T> load_dict_list_##T##_nullable(                      \
    void const* column, i32 row, i32 element, index_type_tag<I> __VA_ARGS__)      \
  {                                                                               \
    auto const& view = *static_cast<input_column const*>(column);                 \
    if (view.is_null(row)) { return {}; }                                         \
    return nullable_list_payload<T, O>(                                           \
      view.child(dictionary_keys_column_index), dict_row<I>(view, row), element); \
  }
#define CUDF_ROW_DICT_ACCESS(T, I, O, ...)          \
  CUDF_ROW_LOAD_DICT(T, I, O, __VA_ARGS__)          \
  CUDF_ROW_LOAD_DICT_LIST(T, I, O, __VA_ARGS__)     \
  CUDF_ROW_SET_DICT(T, I, O, __VA_ARGS__)           \
  CUDF_ROW_SET_DICT_LIST(T, I, O, __VA_ARGS__)      \
  CUDF_ROW_LOAD_DICT_NULLABLE(T, I, O, __VA_ARGS__) \
  CUDF_ROW_LOAD_DICT_LIST_NULLABLE(T, I, O, __VA_ARGS__)
#define CUDF_ROW_STRING_DATA(O)                                               \
  CUDF_ROW_EXPORT char* string_data(void const* column, i32 row, type_tag<O>) \
  {                                                                           \
    return chars<O>(*static_cast<output_column const*>(column), row);         \
  }
#define CUDF_ROW_STRING_SIZE(O)                                             \
  CUDF_ROW_EXPORT i64 string_size(void const* column, i32 row, type_tag<O>) \
  {                                                                         \
    return bytes<O>(*static_cast<output_column const*>(column), row);       \
  }

#ifdef CUDF_ROW_FRAGMENT
CUDF_ROW_FRAGMENT
#else
CUDF_ROW_INPUT_COLUMN()
CUDF_ROW_OUTPUT_COLUMN()
CUDF_ROW_LOAD_LIST_SIZE()
CUDF_ROW_OUTPUT_LIST_SIZE()
CUDF_ROW_SET_STRING_VIEW()
CUDF_ROW_SET_VALIDITY()
CUDF_ROW_SET_LIST_VALIDITY()
CUDF_ROW_LOAD_LIST_SIZE_NULLABLE()
CUDF_ROW_LOAD_DICT_LIST_SIZE(i8)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(i8)
CUDF_ROW_LOAD_DICT_LIST_SIZE(i16)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(i16)
CUDF_ROW_LOAD_DICT_LIST_SIZE(i32)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(i32)
CUDF_ROW_LOAD_DICT_LIST_SIZE(i64)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(i64)
CUDF_ROW_LOAD_DICT_LIST_SIZE(u8)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(u8)
CUDF_ROW_LOAD_DICT_LIST_SIZE(u16)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(u16)
CUDF_ROW_LOAD_DICT_LIST_SIZE(u32)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(u32)
CUDF_ROW_LOAD_DICT_LIST_SIZE(u64)
CUDF_ROW_LOAD_DICT_LIST_SIZE_NULLABLE(u64)
#define CUDF_ROW_INDEX_ACCESS(T, O, ...)       \
  CUDF_ROW_DICT_ACCESS(T, i8, O, __VA_ARGS__)  \
  CUDF_ROW_DICT_ACCESS(T, i16, O, __VA_ARGS__) \
  CUDF_ROW_DICT_ACCESS(T, i32, O, __VA_ARGS__) \
  CUDF_ROW_DICT_ACCESS(T, i64, O, __VA_ARGS__) \
  CUDF_ROW_DICT_ACCESS(T, u8, O, __VA_ARGS__)  \
  CUDF_ROW_DICT_ACCESS(T, u16, O, __VA_ARGS__) \
  CUDF_ROW_DICT_ACCESS(T, u32, O, __VA_ARGS__) \
  CUDF_ROW_DICT_ACCESS(T, u64, O, __VA_ARGS__)
#define CUDF_ROW_ACCESS(T, O, ...) \
  CUDF_ROW_PLAIN_ACCESS(T, O, __VA_ARGS__) CUDF_ROW_INDEX_ACCESS(T, O, __VA_ARGS__)
CUDF_ROW_ACCESS(bool8, i32, )
CUDF_ROW_ACCESS(i8, i32, )
CUDF_ROW_ACCESS(i16, i32, )
CUDF_ROW_ACCESS(i32, i32, )
CUDF_ROW_ACCESS(i64, i32, )
CUDF_ROW_ACCESS(u8, i32, )
CUDF_ROW_ACCESS(u16, i32, )
CUDF_ROW_ACCESS(u32, i32, )
CUDF_ROW_ACCESS(u64, i32, )
CUDF_ROW_ACCESS(f32, i32, )
CUDF_ROW_ACCESS(f64, i32, )
CUDF_ROW_ACCESS(decimal32, i32, )
CUDF_ROW_ACCESS(decimal64, i32, )
CUDF_ROW_ACCESS(decimal128, i32, )
CUDF_ROW_ACCESS(timestamp_days, i32, )
CUDF_ROW_ACCESS(timestamp_seconds, i32, )
CUDF_ROW_ACCESS(timestamp_milliseconds, i32, )
CUDF_ROW_ACCESS(timestamp_microseconds, i32, )
CUDF_ROW_ACCESS(timestamp_nanoseconds, i32, )
CUDF_ROW_ACCESS(duration_days, i32, )
CUDF_ROW_ACCESS(duration_seconds, i32, )
CUDF_ROW_ACCESS(duration_milliseconds, i32, )
CUDF_ROW_ACCESS(duration_microseconds, i32, )
CUDF_ROW_ACCESS(duration_nanoseconds, i32, )
#define CUDF_ROW_OFFSET_TAG(O) , type_tag<O>
CUDF_ROW_ACCESS(string_view, i32, CUDF_ROW_OFFSET_TAG(i32))
CUDF_ROW_ACCESS(string_view, i64, CUDF_ROW_OFFSET_TAG(i64))
CUDF_ROW_STRING_DATA(i32)
CUDF_ROW_STRING_SIZE(i32)
CUDF_ROW_STRING_DATA(i64)
CUDF_ROW_STRING_SIZE(i64)
#undef CUDF_ROW_OFFSET_TAG
#undef CUDF_ROW_ACCESS
#undef CUDF_ROW_INDEX_ACCESS
#undef CUDF_ROW_DICT_ACCESS
#undef CUDF_ROW_PLAIN_ACCESS
#undef CUDF_ROW_EXPORT
#endif
}  // namespace cudf::row_abi
