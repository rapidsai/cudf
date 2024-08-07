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

#include "arrow/array/array_binary.h"
#include "arrow/type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <memory>

// Helper functuons to create StringViews
inline arrow::BinaryViewType::c_type ToInlineStringView(const void* data, int32_t size)
{
  arrow::BinaryViewType::c_type out;
  out.inlined = {size, {}};
  memcpy(&out.inlined.data, data, size);
  return out;
}
inline arrow::BinaryViewType::c_type ToInlineStringView(std::string_view v)
{
  return ToInlineStringView(v.data(), static_cast<int32_t>(v.size()));
}
inline arrow::BinaryViewType::c_type ToStringView(const void* data,
                                                  int32_t size,
                                                  int32_t buffer_index,
                                                  int32_t offset)
{
  if (size <= arrow::BinaryViewType::kInlineSize) { return ToInlineStringView(data, size); }
  arrow::BinaryViewType::c_type out;
  out.ref = {size, {}, buffer_index, offset};
  memcpy(&out.ref.prefix, data, sizeof(out.ref.prefix));
  return out;
}
inline arrow::BinaryViewType::c_type ToStringView(std::string_view v,
                                                  int32_t buffer_index,
                                                  int32_t offset)
{
  return ToStringView(v.data(), static_cast<int32_t>(v.size()), buffer_index, offset);
}

/**
 * @brief Create a StringViewArray
 *
 * @param data_buffers The data buffers
 * @param views The string views
 * @param validate Whether to validate the array
 */
arrow::Result<std::shared_ptr<arrow::StringViewArray>> MakeStringViewArray(
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

/**
 * @brief Convert a vector of strings into a vector of
 * individual chars and a vector of the sizes of the strings
 *
 * @param strings The vector of strings
 */
auto make_chars_and_sizes(std::vector<std::string> strings)
{
  std::vector<char> chars{};
  std::vector<cudf::size_type> sizes;
  for (auto& str : strings) {
    chars.insert(chars.end(), std::cbegin(str), std::cend(str));
    sizes.push_back(str.length());
  }
  return std::make_tuple(std::move(chars), std::move(sizes));
};

/**
 * @brief Convert an Arrow StringViewArray to a cudf column
 *
 * @param array The Arrow StringViewArray
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> ArrowStringViewToCudfColumn(
  std::shared_ptr<arrow::StringViewArray> array,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  // Convert the string views into chars and sizes
  std::vector<std::string> strings;
  for (auto i = 0; i < array->length(); i++) {
    strings.push_back(array->GetString(i));
  }
  auto [chars, sizes] = make_chars_and_sizes(strings);

  // Copy the chars vector the device
  rmm::device_uvector<char> d_chars(chars.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_chars.data(), chars.data(), chars.size() * sizeof(char), cudaMemcpyDefault, stream.value()));

  // Copy the sizes vector to the device
  thrust::device_vector<cudf::size_type> d_sizes(sizes);
  rmm::device_uvector<cudf::size_type> string_sizes(array->length() + 1, stream);
  string_sizes.set_element(0, 0, stream);
  thrust::copy(d_sizes.begin(), d_sizes.end(), string_sizes.begin() + 1);

  // Perform an inclusive scan on the sizes vector to
  // generate the offsets and wrap it as a cudf column
  rmm::device_uvector<cudf::size_type> d_offsets(array->length() + 1, stream);
  thrust::inclusive_scan(
    rmm::exec_policy(stream), string_sizes.begin(), string_sizes.end(), d_offsets.begin());
  auto offsets_col = std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);

  // Create a string column out of the chars and offsets
  return cudf::make_strings_column(
    array->length(), std::move(offsets_col), d_chars.release(), 0, {});
}

int main(int argc, char** argv)
{
  std::vector<std::shared_ptr<arrow::Buffer>> data_buffers;
  std::vector<arrow::BinaryViewType::c_type> views;

  // Define the data buffers
  auto buffer_a = arrow::Buffer::FromString("helloworldapachearrowcudfnvidia");
  data_buffers.push_back(buffer_a);

  // Define the string views
  views.push_back(ToStringView("helloworldapache", 0, 0));
  views.push_back(ToStringView("arrowcudfnvidia", 0, 16));
  views.push_back(ToInlineStringView("nvidia"));

  // Create a StringViewArray
  auto string_view_col = MakeStringViewArray(data_buffers, views, true).ValueOrDie();
  std::cout << string_view_col->ToString() << std::endl;

  // Convert the StringViewArray to a cudf column
  auto cudf_col = ArrowStringViewToCudfColumn(string_view_col);
  std::cout << cudf_col->size() << std::endl;

  return 0;
}
