/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <arrow/api.h>
#include <arrow/util/bitmap_builders.h>

// Creating arrow as per given type_id and buffer arguments
template <typename... Ts>
std::shared_ptr<arrow::Array> to_arrow_array(cudf::type_id id, Ts&&... args)
{
  switch (id) {
    case cudf::type_id::BOOL8:
      return std::make_shared<arrow::BooleanArray>(std::forward<Ts>(args)...);
    case cudf::type_id::INT8: return std::make_shared<arrow::Int8Array>(std::forward<Ts>(args)...);
    case cudf::type_id::INT16:
      return std::make_shared<arrow::Int16Array>(std::forward<Ts>(args)...);
    case cudf::type_id::INT32:
      return std::make_shared<arrow::Int32Array>(std::forward<Ts>(args)...);
    case cudf::type_id::INT64:
      return std::make_shared<arrow::Int64Array>(std::forward<Ts>(args)...);
    case cudf::type_id::UINT8:
      return std::make_shared<arrow::UInt8Array>(std::forward<Ts>(args)...);
    case cudf::type_id::UINT16:
      return std::make_shared<arrow::UInt16Array>(std::forward<Ts>(args)...);
    case cudf::type_id::UINT32:
      return std::make_shared<arrow::UInt32Array>(std::forward<Ts>(args)...);
    case cudf::type_id::UINT64:
      return std::make_shared<arrow::UInt64Array>(std::forward<Ts>(args)...);
    case cudf::type_id::FLOAT32:
      return std::make_shared<arrow::FloatArray>(std::forward<Ts>(args)...);
    case cudf::type_id::FLOAT64:
      return std::make_shared<arrow::DoubleArray>(std::forward<Ts>(args)...);
    case cudf::type_id::TIMESTAMP_DAYS:
      return std::make_shared<arrow::Date32Array>(std::make_shared<arrow::Date32Type>(),
                                                  std::forward<Ts>(args)...);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::SECOND),
                                                     std::forward<Ts>(args)...);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                     std::forward<Ts>(args)...);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::MICRO),
                                                     std::forward<Ts>(args)...);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::NANO),
                                                     std::forward<Ts>(args)...);
    case cudf::type_id::DURATION_SECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::SECOND),
                                                    std::forward<Ts>(args)...);
    case cudf::type_id::DURATION_MILLISECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::MILLI),
                                                    std::forward<Ts>(args)...);
    case cudf::type_id::DURATION_MICROSECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::MICRO),
                                                    std::forward<Ts>(args)...);
    case cudf::type_id::DURATION_NANOSECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::NANO),
                                                    std::forward<Ts>(args)...);
    default: CUDF_FAIL("Unsupported type_id conversion to arrow");
  }
}

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>,
                 std::shared_ptr<arrow::Array>>
get_arrow_array(std::vector<T> const& data, std::vector<uint8_t> const& mask = {})
{
  std::shared_ptr<arrow::Buffer> data_buffer;
  arrow::BufferBuilder buff_builder;
  CUDF_EXPECTS(buff_builder.Append(data.data(), sizeof(T) * data.size()).ok(),
               "Failed to append values");
  CUDF_EXPECTS(buff_builder.Finish(&data_buffer).ok(), "Failed to allocate buffer");

  std::shared_ptr<arrow::Buffer> mask_buffer =
    mask.empty() ? nullptr : arrow::internal::BytesToBits(mask).ValueOrDie();

  return to_arrow_array(cudf::type_to_id<T>(), data.size(), data_buffer, mask_buffer);
}

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>,
                 std::shared_ptr<arrow::Array>>
get_arrow_array(std::initializer_list<T> elements, std::initializer_list<uint8_t> validity = {})
{
  std::vector<T> data(elements);
  std::vector<uint8_t> mask(validity);

  return get_arrow_array<T>(data, mask);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, std::shared_ptr<arrow::Array>> get_arrow_array(
  std::vector<bool> const& data, std::vector<bool> const& mask = {})
{
  std::shared_ptr<arrow::BooleanArray> boolean_array;
  arrow::BooleanBuilder boolean_builder;

  if (mask.empty()) {
    CUDF_EXPECTS(boolean_builder.AppendValues(data).ok(),
                 "Failed to append values to boolean builder");
  } else {
    CUDF_EXPECTS(boolean_builder.AppendValues(data, mask).ok(),
                 "Failed to append values to boolean builder");
  }
  CUDF_EXPECTS(boolean_builder.Finish(&boolean_array).ok(), "Failed to create arrow boolean array");

  return boolean_array;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, std::shared_ptr<arrow::Array>> get_arrow_array(
  std::initializer_list<bool> elements, std::initializer_list<bool> validity = {})
{
  std::vector<bool> mask(validity);
  std::vector<bool> data(elements);

  return get_arrow_array<T>(data, mask);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::shared_ptr<arrow::Array>>
get_arrow_array(std::vector<std::string> const& data, std::vector<uint8_t> const& mask = {})
{
  std::shared_ptr<arrow::StringArray> string_array;
  arrow::StringBuilder string_builder;

  CUDF_EXPECTS(string_builder.AppendValues(data, mask.data()).ok(),
               "Failed to append values to string builder");
  CUDF_EXPECTS(string_builder.Finish(&string_array).ok(), "Failed to create arrow string array");

  return string_array;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::shared_ptr<arrow::Array>>
get_arrow_array(std::initializer_list<std::string> elements,
                std::initializer_list<uint8_t> validity = {})
{
  std::vector<uint8_t> mask(validity);
  std::vector<std::string> data(elements);

  return get_arrow_array<T>(data, mask);
}

template <typename KEY_TYPE, typename IND_TYPE>
std::shared_ptr<arrow::Array> get_arrow_dict_array(std::vector<KEY_TYPE> const& keys,
                                                   std::vector<IND_TYPE> const& ind,
                                                   std::vector<uint8_t> const& validity = {})
{
  auto keys_array    = get_arrow_array<KEY_TYPE>(keys);
  auto indices_array = get_arrow_array<IND_TYPE>(ind, validity);

  return std::make_shared<arrow::DictionaryArray>(
    arrow::dictionary(indices_array->type(), keys_array->type()), indices_array, keys_array);
}

template <typename KEY_TYPE, typename IND_TYPE>
std::shared_ptr<arrow::Array> get_arrow_dict_array(std::initializer_list<KEY_TYPE> keys,
                                                   std::initializer_list<IND_TYPE> ind,
                                                   std::initializer_list<uint8_t> validity = {})
{
  auto keys_array    = get_arrow_array<KEY_TYPE>(keys);
  auto indices_array = get_arrow_array<IND_TYPE>(ind, validity);

  return std::make_shared<arrow::DictionaryArray>(
    arrow::dictionary(indices_array->type(), keys_array->type()), indices_array, keys_array);
}

// Creates only single layered list
template <typename T>
std::shared_ptr<arrow::Array> get_arrow_list_array(std::vector<T> data,
                                                   std::vector<int32_t> offsets,
                                                   std::vector<uint8_t> data_validity = {},
                                                   std::vector<uint8_t> list_validity = {})
{
  auto data_array = get_arrow_array<T>(data, data_validity);
  std::shared_ptr<arrow::Buffer> offset_buffer;
  arrow::BufferBuilder buff_builder;
  CUDF_EXPECTS(buff_builder.Append(offsets.data(), sizeof(int32_t) * offsets.size()).ok(),
               "Failed to append values to buffer builder");
  CUDF_EXPECTS(buff_builder.Finish(&offset_buffer).ok(), "Failed to allocate buffer");

  auto nullable = std::accumulate(list_validity.begin(), list_validity.end(), 0) > 0;
  return std::make_shared<arrow::ListArray>(
    arrow::list(arrow::field("", data_array->type(), nullable)),
    offsets.size() - 1,
    offset_buffer,
    data_array,
    list_validity.empty() ? nullptr : arrow::internal::BytesToBits(list_validity).ValueOrDie());
}

template <typename T>
std::shared_ptr<arrow::Array> get_arrow_list_array(
  std::initializer_list<T> data,
  std::initializer_list<int32_t> offsets,
  std::initializer_list<uint8_t> data_validity = {},
  std::initializer_list<uint8_t> list_validity = {})
{
  std::vector<T> data_vector(data);
  std::vector<int32_t> ofst(offsets);
  std::vector<uint8_t> data_mask(data_validity);
  std::vector<uint8_t> list_mask(list_validity);
  return get_arrow_list_array<T>(data_vector, ofst, data_mask, list_mask);
}

std::pair<std::unique_ptr<cudf::table>, std::shared_ptr<arrow::Table>> get_tables(
  cudf::size_type length = 10000);

template <typename T>
[[nodiscard]] auto make_decimal128_arrow_array(std::vector<T> const& data,
                                               std::optional<std::vector<int>> const& validity,
                                               int32_t scale) -> std::shared_ptr<arrow::Array>
{
  auto constexpr BIT_WIDTH_RATIO = sizeof(__int128_t) / sizeof(T);

  std::shared_ptr<arrow::Array> arr;
  arrow::Decimal128Builder decimal_builder(arrow::decimal(cudf::detail::max_precision<T>(), -scale),
                                           arrow::default_memory_pool());

  for (T i = 0; i < static_cast<T>(data.size() / BIT_WIDTH_RATIO); ++i) {
    if (validity.has_value() and not validity.value()[i]) {
      CUDF_EXPECTS(decimal_builder.AppendNull().ok(), "Failed to append");
    } else {
      CUDF_EXPECTS(
        decimal_builder.Append(reinterpret_cast<uint8_t const*>(data.data() + BIT_WIDTH_RATIO * i))
          .ok(),
        "Failed to append");
    }
  }

  CUDF_EXPECTS(decimal_builder.Finish(&arr).ok(), "Failed to build array");

  return arr;
}
