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

#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/csv.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <arrow/array/array_binary.h>
#include <arrow/type.h>

// Helper functuons to create StringViews
inline arrow::BinaryViewType::c_type to_inline_string_view(const void* data, int32_t const& size)
{
  arrow::BinaryViewType::c_type out;
  out.inlined = {size, {}};
  memcpy(&out.inlined.data, data, size);
  return out;
}
inline arrow::BinaryViewType::c_type to_inline_string_view(std::string_view const& v)
{
  return to_inline_string_view(v.data(), static_cast<int32_t>(v.size()));
}
inline arrow::BinaryViewType::c_type to_string_view(const void* data,
                                                    int32_t const& size,
                                                    int32_t const& buffer_index,
                                                    int32_t const& offset)
{
  if (size <= arrow::BinaryViewType::kInlineSize) { return to_inline_string_view(data, size); }
  arrow::BinaryViewType::c_type out;
  out.ref = {size, {}, buffer_index, offset};
  memcpy(&out.ref.prefix, data, sizeof(out.ref.prefix));
  return out;
}
inline arrow::BinaryViewType::c_type to_string_view(std::string_view const& v,
                                                    int32_t const& buffer_index,
                                                    int32_t const& offset)
{
  return to_string_view(v.data(), static_cast<int32_t>(v.size()), buffer_index, offset);
}

/**
 * @brief Create a StringViewArray
 *
 * @param data_buffers The data buffers
 * @param views The string views
 * @param validate Whether to validate the array
 */
arrow::Result<std::shared_ptr<arrow::StringViewArray>> MakeStringViewArray(
  arrow::BufferVector const& data_buffers,
  std::vector<arrow::BinaryViewType::c_type> const& views,
  bool validate = true)
{
  auto const length = static_cast<int64_t>(views.size());
  auto const arr    = std::make_shared<arrow::StringViewArray>(
    arrow::utf8_view(), length, arrow::Buffer::FromVector(views), std::move(data_buffers));
  if (validate) { RETURN_NOT_OK(arr->ValidateFull()); }
  return arr;
}

/**
 * @brief Convert a vector of strings into a vector of the
 * constituent chars and a vector of offsets.
 *
 * @param strings The vector of strings
 */
auto make_chars_and_offsets(std::vector<std::string> const& strings)
{
  std::vector<char> chars{};
  std::vector<cudf::size_type> offsets(1, 0);
  for (auto& str : strings) {
    chars.insert(chars.end(), std::cbegin(str), std::cend(str));
    auto const last_offset = static_cast<std::size_t>(offsets.back());
    auto const next_offset = last_offset + str.length();
    CUDF_EXPECTS(
      next_offset < static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
      "Cannot use ArrowStringViewToCudfColumn to build a large strings column");
    offsets.push_back(static_cast<cudf::size_type>(next_offset));
  }
  return std::make_tuple(std::move(chars), std::move(offsets));
};

/**
 * @brief Convert an Arrow StringViewArray to a cudf::column
 *
 * @param array The Arrow StringViewArray
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> ArrowStringViewToCudfColumn(
  std::shared_ptr<arrow::StringViewArray> const& array,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  // Convert the string views into chars and offsets
  std::vector<std::string> strings;
  for (auto i = 0; i < array->length(); i++) {
    strings.push_back(array->GetString(i));
  }
  auto const [chars, offsets] = make_chars_and_offsets(strings);

  // Copy the chars vector to the device
  rmm::device_uvector<char> d_chars(chars.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_chars.data(), chars.data(), chars.size() * sizeof(char), cudaMemcpyDefault, stream.value()));

  // Copy the offsets vector to the device
  // and wrap it in a cudf::column
  rmm::device_uvector<cudf::size_type> d_offsets(offsets.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                                offsets.data(),
                                offsets.size() * sizeof(cudf::size_type),
                                cudaMemcpyDefault,
                                stream.value()));
  auto offsets_col = std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);

  // Create a string column out of the chars and offsets
  return cudf::make_strings_column(
    array->length(), std::move(offsets_col), d_chars.release(), 0, {});
}

int main(int argc, char** argv)
{
  std::vector<std::shared_ptr<arrow::Buffer>> data_buffers;
  std::vector<arrow::BinaryViewType::c_type> views;

  // Define the data buffers and string views
  auto const buffer_a =
    arrow::Buffer::FromString("hello rapids teamapache arrow interopnvidiacudf");
  data_buffers.push_back(buffer_a);
  views.push_back(to_string_view("hello rapid steam", 0, 0));
  views.push_back(to_string_view("apache arrow interop", 0, 17));
  views.push_back(to_inline_string_view("nvidia"));
  views.push_back(to_inline_string_view("cudf"));

  // Create a StringViewArray
  auto const string_view_col = MakeStringViewArray(data_buffers, views, true).ValueOrDie();
  std::cout << string_view_col->ToString() << std::endl;

  // Convert the StringViewArray to a cudf::column
  auto const cudf_col = ArrowStringViewToCudfColumn(string_view_col);

  // Write the cudf::column as CSV
  auto const tbl_view                  = cudf::table_view({cudf_col->view()});
  std::vector<std::string> const names = {"col_a"};

  std::vector<char> h_buffer;
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(cudf::io::sink_info(&h_buffer), tbl_view)
      .include_header(not names.empty())
      .names(names);

  cudf::io::write_csv(writer_options);
  auto const result = std::string(h_buffer.data(), h_buffer.size());
  std::cout << result << std::endl;

  return 0;
}
