/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {
namespace {
struct byte_list_conversion {
  /**
   * @brief Function object for converting primitive types and string columns to lists of bytes.
   */
  template <typename T>
  std::enable_if_t<!std::is_integral_v<T> and !is_floating_point<T>(), std::unique_ptr<column>>
  operator()(column_view const&,
             flip_endianness,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Unsupported non-numeric and non-string column");
  }

  template <typename T>
  std::enable_if_t<is_floating_point<T>() or std::is_integral_v<T>, std::unique_ptr<column>>
  operator()(column_view const& input_column,
             flip_endianness configuration,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    size_type num_bytes = input_column.size() * sizeof(T);
    auto byte_column    = make_numeric_column(
      data_type{type_id::UINT8}, num_bytes, mask_state::UNALLOCATED, stream, mr);

    char* d_chars      = reinterpret_cast<char*>(byte_column->mutable_view().data<uint8_t>());
    char const* d_data = reinterpret_cast<char const*>(input_column.data<T>());
    size_type mask     = sizeof(T) - 1;

    if (configuration == flip_endianness::YES) {
      thrust::for_each(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_bytes),
                       [d_chars, d_data, mask] __device__(auto index) {
                         d_chars[index] = d_data[index + mask - ((index & mask) << 1)];
                       });
    } else {
      thrust::copy_n(rmm::exec_policy(stream), d_data, num_bytes, d_chars);
    }

    auto begin          = thrust::make_constant_iterator(cudf::size_of(input_column.type()));
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      begin, begin + input_column.size(), stream, mr);

    rmm::device_buffer null_mask = detail::copy_bitmask(input_column, stream, mr);

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
  flip_endianness,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  strings_column_view input_strings(input_column);
  auto strings_count = input_strings.size();
  if (strings_count == 0) return cudf::empty_like(input_column);

  auto col_content = std::make_unique<column>(input_column, stream, mr)->release();
  auto contents =
    col_content.children[strings_column_view::chars_column_index].release()->release();
  auto data      = contents.data.release();
  auto null_mask = contents.null_mask.release();
  auto uint8_col = std::make_unique<column>(data_type{type_id::UINT8},
                                            data->size(),
                                            std::move(*data),
                                            std::move(*null_mask),
                                            UNKNOWN_NULL_COUNT);

  return make_lists_column(
    input_column.size(),
    std::move(col_content.children[cudf::strings_column_view::offsets_column_index]),
    std::move(uint8_col),
    input_column.null_count(),
    detail::copy_bitmask(input_column, stream, mr),
    stream,
    mr);
}
}  // namespace

/**
 * @copydoc cudf::byte_cast(column_view const&, flip_endianness, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> byte_cast(column_view const& input_column,
                                  flip_endianness endian_configuration,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input_column.type(), byte_list_conversion{}, input_column, endian_configuration, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::byte_cast(column_view const&, flip_endianness, rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> byte_cast(column_view const& input_column,
                                  flip_endianness endian_configuration,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_cast(input_column, endian_configuration, cudf::default_stream_value, mr);
}

}  // namespace cudf
