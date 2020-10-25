/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <arrow/util/bitmap_builders.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#pragma once

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same<T, bool>::value,
                 std::shared_ptr<arrow::Array>>
get_arrow_array(std::vector<T> const& data, std::vector<uint8_t> const& mask = {})
{
  std::shared_ptr<arrow::Buffer> data_buffer;
  arrow::BufferBuilder buff_builder;
  buff_builder.Append(data.data(), sizeof(T) * data.size());
  CUDF_EXPECTS(buff_builder.Finish(&data_buffer).ok(), "Failed to allocate buffer");

  std::shared_ptr<arrow::Buffer> mask_buffer =
    mask.empty() ? nullptr : arrow::internal::BytesToBits(mask).ValueOrDie();

  return cudf::detail::to_arrow_array(cudf::type_to_id<T>(), data.size(), data_buffer, mask_buffer);
}

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same<T, bool>::value,
                 std::shared_ptr<arrow::Array>>
get_arrow_array(std::initializer_list<T> elements, std::initializer_list<uint8_t> validity = {})
{
  std::vector<T> data(elements);
  std::vector<uint8_t> mask(validity);

  return get_arrow_array<T>(data, mask);
}

template <typename T>
std::enable_if_t<std::is_same<T, bool>::value, std::shared_ptr<arrow::Array>> get_arrow_array(
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
std::enable_if_t<std::is_same<T, bool>::value, std::shared_ptr<arrow::Array>> get_arrow_array(
  std::initializer_list<bool> elements, std::initializer_list<bool> validity = {})
{
  std::vector<bool> mask(validity);
  std::vector<bool> data(elements);

  return get_arrow_array<T>(data, mask);
}

template <typename T>
std::enable_if_t<std::is_same<T, cudf::string_view>::value, std::shared_ptr<arrow::Array>>
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
std::enable_if_t<std::is_same<T, cudf::string_view>::value, std::shared_ptr<arrow::Array>>
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

  return std::make_shared<arrow::ListArray>(
    arrow::list(data_array->type()),
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
