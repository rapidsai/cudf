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
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop/detail/arrow.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <nanoarrow/nanoarrow.hpp>

// no-op allocator/deallocator to set into ArrowArray buffers that we don't
// want to own their buffers.
static ArrowBufferAllocator noop_alloc = (struct ArrowBufferAllocator){
  .reallocate = [](ArrowBufferAllocator*, uint8_t* ptr, int64_t, int64_t) -> uint8_t* {
    return ptr;
  },
  .free         = [](ArrowBufferAllocator*, uint8_t*, int64_t) {},
  .private_data = nullptr,
};

// populate the ArrowArray by copying host data buffers for fixed width types other
// than boolean.
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

// populate an ArrowArray with pointers to the raw device buffers of a cudf::column_view
// and use the no-op alloc so that the ArrowArray doesn't presume ownership of the data
template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>() and !std::is_same_v<T, bool>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
  ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(T) * view.size();
  ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(view.data<uint8_t>());
}

// populate an ArrowArray with boolean data by generating the appropriate
// bitmaps to copy the data.
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

// populate an ArrowArray from a boolean cudf column. Since Arrow and cudf
// still represent boolean arrays differently, we have to use bools_to_mask
// and give the ArrowArray object ownership of the device data.
template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, void> populate_from_col(ArrowArray* arr,
                                                                  cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));

  auto bitmask = cudf::bools_to_mask(view);
  auto ptr     = reinterpret_cast<uint8_t*>(bitmask.first->data());
  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(
    ArrowArrayBuffer(arr, 1),
    ArrowBufferDeallocator(
      [](ArrowBufferAllocator* alloc, uint8_t*, int64_t) {
        auto buf = reinterpret_cast<std::unique_ptr<rmm::device_buffer>*>(alloc->private_data);
        delete buf;
      },
      new std::unique_ptr<rmm::device_buffer>(std::move(bitmask.first)))));
  ArrowArrayBuffer(arr, 1)->size_bytes = cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayBuffer(arr, 1)->data       = ptr;
}

// populate an ArrowArray by copying the string data and constructing the offsets
// buffer.
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

// populate an ArrowArray with the string data buffers of a cudf column_view
// using no-op allocator so the ArrowArray knows it doesn't have ownership
// of the device buffers.
template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> populate_from_col(
  ArrowArray* arr, cudf::column_view view)
{
  arr->length     = view.size();
  arr->null_count = view.null_count();

  NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc));
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(view.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view.null_mask()));

  cudf::strings_column_view sview{view};
  if (view.size() > 0) {
    NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc));
    ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(int32_t) * sview.offsets().size();
    ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(sview.offsets().data<uint8_t>());
    NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 2), noop_alloc));
    ArrowArrayBuffer(arr, 2)->size_bytes = sview.chars_size(cudf::get_default_stream());
    ArrowArrayBuffer(arr, 2)->data       = const_cast<uint8_t*>(view.data<uint8_t>());
  } else {
    auto zero          = rmm::device_scalar<int32_t>(0, cudf::get_default_stream());
    const uint8_t* ptr = reinterpret_cast<uint8_t*>(zero.data());
    nanoarrow::BufferInitWrapped(ArrowArrayBuffer(arr, 1), std::move(zero), ptr, 4);
  }
}

// populate a dictionary ArrowArray by delegating the copying of the indices
// and key arrays
template <typename KEY_TYPE, typename IND_TYPE>
void get_nanoarrow_dict_array(ArrowArray* arr,
                              std::vector<KEY_TYPE> const& keys,
                              std::vector<IND_TYPE> const& ind,
                              std::vector<uint8_t> const& validity = {})
{
  get_nanoarrow_array<KEY_TYPE>(arr->dictionary, keys);
  get_nanoarrow_array<IND_TYPE>(arr, ind, validity);
}

template <typename KEY_TYPE, typename IND_TYPE>
void populate_dict_from_col(ArrowArray* arr, cudf::dictionary_column_view dview)
{
  arr->length     = dview.size();
  arr->null_count = dview.null_count();
  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 0), noop_alloc);
  ArrowArrayValidityBitmap(arr)->buffer.size_bytes =
    cudf::bitmask_allocation_size_bytes(dview.size());
  ArrowArrayValidityBitmap(arr)->buffer.data =
    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(dview.null_mask()));

  ArrowBufferSetAllocator(ArrowArrayBuffer(arr, 1), noop_alloc);
  ArrowArrayBuffer(arr, 1)->size_bytes = sizeof(IND_TYPE) * dview.indices().size();
  ArrowArrayBuffer(arr, 1)->data       = const_cast<uint8_t*>(dview.indices().data<uint8_t>());

  populate_from_col<KEY_TYPE>(arr->dictionary, dview.keys());
}

// populate a list ArrowArray by copying the offsets and data buffers
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

std::tuple<std::unique_ptr<cudf::table>, nanoarrow::UniqueSchema, nanoarrow::UniqueArray>
get_nanoarrow_tables(cudf::size_type length = 10000);

void populate_list_from_col(ArrowArray* arr, cudf::lists_column_view view);
