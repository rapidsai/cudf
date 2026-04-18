
#pragma once
#include <cudf/column/column_device_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <jit/element_storage.cuh>
#include <jit/lite/lite.cuh>

namespace cudf {

__device__ void load_element(bool is_scalar,
                             column_device_view const* column,
                             bitmask_type const* __restrict__ null_mask,
                             size_type element_index,
                             element_storage* storage)
{
  auto index = is_scalar ? 0 : element_index;

#define CUDF_CASE(TYPE_ID, type_name)                                                            \
  case type_id::TYPE_ID: {                                                                       \
    static_assert(sizeof(type_name) <= sizeof(element_storage));                                 \
    static_assert(alignof(type_name) <= alignof(element_storage));                               \
    static_assert(sizeof(cuda::std::optional<type_name>) <= sizeof(element_storage));            \
    static_assert(alignof(cuda::std::optional<type_name>) <= alignof(element_storage));          \
                                                                                                 \
    if (!column->nullable()) {                                                                   \
      auto v                                       = column->template element<type_name>(index); \
      *reinterpret_cast<type_name*>(storage->data) = v;                                          \
    } else {                                                                                     \
      auto v = column->template nullable_element<type_name>(index);                              \
      *reinterpret_cast<cuda::std::optional<type_name>*>(storage->data) = v;                     \
    }                                                                                            \
  } break;

  switch (column->type().id()) {
    CUDF_CASE(INT8, int8_t)
    CUDF_CASE(INT16, int16_t)
    CUDF_CASE(INT32, int32_t)
    CUDF_CASE(INT64, int64_t)
    CUDF_CASE(UINT8, uint8_t)
    CUDF_CASE(UINT16, uint16_t)
    CUDF_CASE(UINT32, uint32_t)
    CUDF_CASE(UINT64, uint64_t)
    CUDF_CASE(FLOAT32, float)
    CUDF_CASE(FLOAT64, double)
    CUDF_CASE(BOOL8, bool)
    CUDF_CASE(DECIMAL32, numeric::decimal32)
    CUDF_CASE(DECIMAL64, numeric::decimal64)
    CUDF_CASE(DECIMAL128, numeric::decimal128)
    CUDF_CASE(TIMESTAMP_DAYS, cudf::timestamp_D)
    CUDF_CASE(TIMESTAMP_SECONDS, cudf::timestamp_s)
    CUDF_CASE(TIMESTAMP_MILLISECONDS, cudf::timestamp_ms)
    CUDF_CASE(TIMESTAMP_MICROSECONDS, cudf::timestamp_us)
    CUDF_CASE(TIMESTAMP_NANOSECONDS, cudf::timestamp_ns)
    CUDF_CASE(DURATION_DAYS, cudf::duration_D)
    CUDF_CASE(DURATION_SECONDS, cudf::duration_s)
    CUDF_CASE(DURATION_MILLISECONDS, cudf::duration_ms)
    CUDF_CASE(DURATION_MICROSECONDS, cudf::duration_us)
    CUDF_CASE(DURATION_NANOSECONDS, cudf::duration_ns)
    CUDF_CASE(STRING, cudf::string_view)
    default: CUDF_UNREACHABLE();
  }

#undef CUDF_CASE
}

__device__ void store_element(mutable_column_device_view const* column,
                              element_storage const* storage,
                              size_type element_index,
                              unsigned int active_mask)
{
// TODO: compact validity into a bitmask
// (warp_compact_validity<A>(active_mask, output_cols, element_idx, is_valid[A::index]), ...);
#define CUDF_CASE(TYPE_ID, type_name)                                                   \
  case type_id::TYPE_ID: {                                                              \
    static_assert(sizeof(type_name) <= sizeof(element_storage));                        \
    static_assert(alignof(type_name) <= alignof(element_storage));                      \
    static_assert(sizeof(cuda::std::optional<type_name>) <= sizeof(element_storage));   \
    static_assert(alignof(cuda::std::optional<type_name>) <= alignof(element_storage)); \
                                                                                        \
    if (!column->nullable()) {                                                          \
      auto v = *reinterpret_cast<type_name const*>(storage->data);                      \
      column->template assign<type_name>(element_index, v);                             \
    } else {                                                                            \
      auto v = *reinterpret_cast<cuda::std::optional<type_name> const*>(storage->data); \
      column->template assign<type_name>(element_index, *v);                            \
    }                                                                                   \
  } break;

  switch (column->type().id()) {
    CUDF_CASE(INT16, int16_t)
    CUDF_CASE(INT32, int32_t)
    CUDF_CASE(INT64, int64_t)
    CUDF_CASE(UINT8, uint8_t)
    CUDF_CASE(UINT16, uint16_t)
    CUDF_CASE(UINT32, uint32_t)
    CUDF_CASE(UINT64, uint64_t)
    CUDF_CASE(FLOAT32, float)
    CUDF_CASE(FLOAT64, double)
    CUDF_CASE(BOOL8, bool)
    CUDF_CASE(DECIMAL32, numeric::decimal32)
    CUDF_CASE(DECIMAL64, numeric::decimal64)
    CUDF_CASE(DECIMAL128, numeric::decimal128)
    CUDF_CASE(TIMESTAMP_DAYS, cudf::timestamp_D)
    CUDF_CASE(TIMESTAMP_SECONDS, cudf::timestamp_s)
    CUDF_CASE(TIMESTAMP_MILLISECONDS, cudf::timestamp_ms)
    CUDF_CASE(TIMESTAMP_MICROSECONDS, cudf::timestamp_us)
    CUDF_CASE(TIMESTAMP_NANOSECONDS, cudf::timestamp_ns)
    CUDF_CASE(DURATION_DAYS, cudf::duration_D)
    CUDF_CASE(DURATION_SECONDS, cudf::duration_s)
    CUDF_CASE(DURATION_MILLISECONDS, cudf::duration_ms)
    CUDF_CASE(DURATION_MICROSECONDS, cudf::duration_us)
    CUDF_CASE(DURATION_NANOSECONDS, cudf::duration_ns)
    default: CUDF_UNREACHABLE();
  }
}

}  // namespace cudf
