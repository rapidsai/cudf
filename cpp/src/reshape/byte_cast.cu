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

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace {
struct byte_list_conversion {
  /**
   * @brief Function object for converting primitive types and string columns to lists of bytes.
   */
  template <typename T>
  std::enable_if_t<!std::is_integral<T>::value and !is_floating_point<T>(), std::unique_ptr<column>>
  operator()(column_view const& input_column,
             flip_endianness configuration,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) const
  {
    CUDF_FAIL("Unsupported non-numeric and non-string column");
  }

  template <typename T>
  std::enable_if_t<is_floating_point<T>() or std::is_integral<T>::value, std::unique_ptr<column>>
  operator()(column_view const& input_column,
             flip_endianness configuration,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) const
  {
    size_type num_bytes = input_column.size() * sizeof(T);
    auto byte_column    = make_numeric_column(
      data_type{type_id::UINT8}, num_bytes, mask_state::UNALLOCATED, stream, mr);

    char* d_chars      = reinterpret_cast<char*>(byte_column->mutable_view().data<uint8_t>());
    char const* d_data = reinterpret_cast<char const*>(input_column.data<T>());
    size_type mask     = sizeof(T) - 1;

    if (configuration == flip_endianness::YES) {
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_bytes),
                       [d_chars, d_data, mask] __device__(auto index) {
                         d_chars[index] = d_data[index + mask - ((index & mask) << 1)];
                       });
    } else {
      thrust::copy_n(rmm::exec_policy(stream)->on(stream), d_data, num_bytes, d_chars);
    }

    auto begin          = thrust::make_constant_iterator(cudf::size_of(input_column.type()));
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      begin, begin + input_column.size(), mr, stream);

    rmm::device_buffer null_mask = copy_bitmask(input_column, stream, mr);

    return make_lists_column(input_column.size(),
                             std::move(offsets_column),
                             std::move(byte_column),
                             input_column.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
  }
};

template <>
std::unique_ptr<cudf::column> byte_list_conversion::operator()<string_view>(
  column_view const& input_column,
  flip_endianness configuration,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  strings_column_view input_strings(input_column);
  auto strings_count = input_strings.size();
  if (strings_count == 0) return cudf::empty_like(input_column);

  auto contents = std::make_unique<column>(input_column, stream, mr)->release();
  return make_lists_column(
    input_column.size(),
    std::move(contents.children[cudf::strings_column_view::offsets_column_index]),
    std::move(contents.children[cudf::strings_column_view::chars_column_index]),
    input_column.null_count(),
    copy_bitmask(input_column, stream, mr),
    stream,
    mr);
}
}  // namespace

std::unique_ptr<column> byte_cast(column_view const& input_column,
                                  flip_endianness configuration,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  CUDF_FUNC_RANGE();
  return type_dispatcher(
    input_column.type(), byte_list_conversion{}, input_column, configuration, mr, stream);
}

}  // namespace cudf
