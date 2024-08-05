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

#include <cudf/interop.hpp>

inline arrow::BinaryViewType::c_type ToInlineBinaryView(const void* data, int32_t size)
{
  // Small string: inlined. Bytes beyond size are zeroed
  arrow::BinaryViewType::c_type out;
  out.inlined = {size, {}};
  memcpy(&out.inlined.data, data, size);
  return out;
}

inline arrow::BinaryViewType::c_type ToInlineBinaryView(std::string_view v)
{
  return ToInlineBinaryView(v.data(), static_cast<int32_t>(v.size()));
}

inline arrow::BinaryViewType::c_type ToBinaryView(const void* data,
                                                  int32_t size,
                                                  int32_t buffer_index,
                                                  int32_t offset)
{
  if (size <= arrow::BinaryViewType::kInlineSize) { return ToInlineBinaryView(data, size); }

  // Large string: store index/offset.
  arrow::BinaryViewType::c_type out;
  out.ref = {size, {}, buffer_index, offset};
  memcpy(&out.ref.prefix, data, sizeof(out.ref.prefix));
  return out;
}

inline arrow::BinaryViewType::c_type ToBinaryView(std::string_view v,
                                                  int32_t buffer_index,
                                                  int32_t offset)
{
  return ToBinaryView(v.data(), static_cast<int32_t>(v.size()), buffer_index, offset);
}

arrow::Result<std::shared_ptr<arrow::StringViewArray>> MakeBinaryViewArray(
  arrow::BufferVector data_buffers,
  const std::vector<arrow::BinaryViewType::c_type>& views,
  bool validate = true)
{
  auto length = static_cast<int64_t>(views.size());
  auto arr    = std::make_shared<arrow::StringViewArray>(
    arrow::utf8_view(), length, arrow::Buffer::FromVector(views), std::move(data_buffers));
  if (validate) { RETURN_NOT_OK(arr->ValidateFull()); }
  return arr;
}

int main(int argc, char** argv)
{
  std::vector<std::shared_ptr<arrow::Buffer>> data_buffers;
  std::vector<arrow::BinaryViewType::c_type> views;

  auto buffer_a = arrow::Buffer::FromString("helloworldapachearrowcudfnvidia");
  data_buffers.push_back(buffer_a);

  views.push_back(ToBinaryView("helloworld", 0, 0));
  views.push_back(ToBinaryView("apachearrow", 0, 10));
  views.push_back(ToInlineBinaryView("cudf"));

  auto string_view_col = MakeBinaryViewArray(data_buffers, views, true).ValueOrDie();
  for (int i = 0; i < string_view_col->length(); i++) {
    std::cout << string_view_col->GetString(i) << std::endl;
  }

  return 0;
}
