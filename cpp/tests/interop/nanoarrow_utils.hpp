/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/interop/detail/arrow.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

static ArrowBufferAllocator noop_alloc = (struct ArrowBufferAllocator){
  .reallocate = [](ArrowBufferAllocator*, uint8_t* ptr, int64_t, int64_t) -> uint8_t* {
    return ptr;
  },
  .free         = [](ArrowBufferAllocator*, uint8_t*, int64_t) {},
  .private_data = nullptr,
};

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>, void> get_nanoarrow_array(
  ArrowArray* arr, std::vector<T> const& data, std::vector<uint8_t> const& mask = {})
{
  arr->length = data.size();
  NANOARROW_THROW_NOT_OK(
    ArrowBufferAppend(ArrowArrayBuffer(arr, 1), data.data(), sizeof(T) * data.size()));
  if (!mask.empty()) {
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(ArrowArrayValidityBitmap(arr), mask.size()));
    ArrowBitmapAppendInt8Unsafe(
      ArrowArrayValidityBitmap(arr), reinterpret_cast<const int8_t*>(mask.data()), mask.size());
    arr->null_count = ArrowBitCountSet(ArrowArrayValidityBitmap(arr)->buffer.data, 0, data.size());
  } else {
    arr->null_count = 0;
  }

  CUDF_EXPECTS(ArrowArrayFinishBuildingDefault(arr, nullptr) == NANOARROW_OK,
               "failed to construct array");
}

template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc);
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc);
  ArrowArrayBuffer(arr, 1)->data = const_cast<uint8_t*>(view.data<uint8_t>());
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, void> get_nanoarrow_array(
  ArrowArray* arr, std::vector<bool> const& data, std::vector<bool> const& mask = {})
{
  ArrowBitmap bool_data;
  ArrowBitmapInit(&bool_data);
  NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(&bool_data, data.size()));
  std::for_each(data.begin(), data.end(), [&](const auto&& elem) {
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(&bool_data, (elem) ? 1 : 0, 1));
  });
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(arr, 1, &bool_data.buffer));

  if (!mask.empty()) {
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(ArrowArrayValidityBitmap(arr), mask.size()));
    std::for_each(mask.begin(), mask.end(), [&](const auto&& elem) {
      NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(arr), (elem) ? 1 : 0, 1));
    });
    arr->null_count = ArrowBitCountSet(ArrowArrayValidityBitmap(arr)->buffer.data, 0, data.size());
  } else {
    arr->null_count = 0;
  }

  CUDF_EXPECTS(ArrowArrayFinishBuildingDefault(arr, nullptr) == NANOARROW_OK,
               "failed to construct boolean array");
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, void> populate_from_col(ArrowArray* arr,
                                                                  cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc);
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));

  auto bitmask = cudf::bools_to_mask(view);
  auto ptr     = reinterpret_cast<uint8_t*>(bitmask.first->data());
  ArrowBufferSetAllocator(
    ArrowArrayBuffer(arr, 1),
    ArrowBufferDeallocator(
      [](ArrowBufferAllocator* alloc, uint8_t*, int64_t) {
        auto buf = reinterpret_cast<std::unique_ptr<rmm::device_buffer>*>(alloc->private_data);
        delete buf;
      },
      new std::unique_ptr<rmm::device_buffer>(std::move(bitmask.first))));
  ArrowArrayBuffer(arr, 1)->data = ptr;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> get_nanoarrow_array(
  ArrowArray* arr, std::vector<std::string> const& data, std::vector<uint8_t> const& mask = {})
{
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(arr));
  for (auto& str : data) {
    NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(arr, ArrowCharView(str.c_str())));
  }

  if (!mask.empty()) {
    ArrowBitmapReset(ArrowArrayValidityBitmap(arr));
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(ArrowArrayValidityBitmap(arr), mask.size()));
    ArrowBitmapAppendInt8Unsafe(
      ArrowArrayValidityBitmap(arr), reinterpret_cast<const int8_t*>(mask.data()), mask.size());
    arr->null_count = ArrowBitCountSet(ArrowArrayValidityBitmap(arr)->buffer.data, 0, data.size());
  } else {
    arr->null_count = 0;
  }

  CUDF_EXPECTS(ArrowArrayFinishBuildingDefault(arr, nullptr) == NANOARROW_OK,
               "failed to construct string array");
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc);
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));

  cudf::strings_column_view sview{view};
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc);
  ArrowArrayBuffer(arr, 1)->data = const_cast<uint8_t*>(sview.offsets().data<uint8_t>());
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 2), noop_alloc);
  ArrowArrayBuffer(arr, 2)->data = const_cast<uint8_t*>(view.data<uint8_t>());
}

template <typename KEY_TYPE, typename IND_TYPE>
void get_nanoarrow_dict_array(ArrowArray* arr,
                              std::vector<KEY_TYPE> const& keys,
                              std::vector<IND_TYPE> const& ind,
                              std::vector<uint8_t> const& validity = {})
{
  get_nanoarrow_array<KEY_TYPE>(arr->dictionary, keys);
  get_nanoarrow_array<IND_TYPE>(arr, ind, validity);
}

template <typename T>
void get_nanoarrow_list_array(ArrowArray* arr,
                              std::vector<T> data,
                              std::vector<int32_t> offsets,
                              std::vector<uint8_t> data_validity = {},
                              std::vector<uint8_t> list_validity = {})
{
  get_nanoarrow_array<T>(arr->children[0], data, data_validity);

  arr->length = offsets.size() - 1;
  NANOARROW_THROW_NOT_OK(
    ArrowBufferAppend(ArrowArrayBuffer(arr, 1), offsets.data(), sizeof(int32_t) * offsets.size()));
  if (!list_validity.empty()) {
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(ArrowArrayValidityBitmap(arr), list_validity.size()));
    ArrowBitmapAppendInt8Unsafe(ArrowArrayValidityBitmap(arr),
                                reinterpret_cast<const int8_t*>(list_validity.data()),
                                arr->length);
    arr->null_count = ArrowBitCountSet(ArrowArrayValidityBitmap(arr)->buffer.data, 0, arr->length);
  } else {
    arr->null_count = 0;
  }

  CUDF_EXPECTS(ArrowArrayFinishBuildingDefault(arr, nullptr) == NANOARROW_OK,
               "failed to construct list array");
}

void populate_list_from_col(ArrowArray* arr, cudf::lists_column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc);
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));

  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc);
  ArrowArrayBuffer(arr, 1)->data = const_cast<uint8_t*>(view.offsets().data<uint8_t>());
}
