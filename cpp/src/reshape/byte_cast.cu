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

#include <algorithm>
#include <cstdint>
#include <cudf/strings/detail/utilities.hpp>
#include <memory>

#include "cudf/column/column_device_view.cuh"
#include "cudf/detail/nvtx/ranges.hpp"
#include "cudf/replace.hpp"
#include "cudf/strings/detail/utilities.cuh"
#include "cudf/types.hpp"
#include "cudf/utilities/traits.hpp"
#include "cudf/utilities/type_dispatcher.hpp"

namespace cudf {
namespace detail {
struct ByteListConversion {
  /**
   * @brief Function object for converting primitive types and string columns to lists of bytes,
   * mimics Spark's cast to binary type.
   */
  struct flip_endianness {
    char* d_chars;
    const char* d_data;
    uint32_t mask;
    __device__ void operator()(int byte_index)
    {
      d_chars[byte_index] = d_data[byte_index + mask - ((byte_index & mask) << 1)];
    }
  };

  template <
    typename T,
    typename std::enable_if_t<!std::is_integral<T>::value and !is_floating_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     endianess_policy configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    release_assert(false && "MD5 Unsupported non-numeric and non-string column");
    return NULL;
  }

  template <typename T, typename std::enable_if_t<is_floating_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     endianess_policy configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    size_type num_output_elements = input_column.size() * cudf::size_of(input_column.type());

    auto begin          = thrust::make_constant_iterator(cudf::size_of(input_column.type()));
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      begin, begin + input_column.size(), mr, stream);

    auto byte_column = make_numeric_column(
      data_type{type_id::UINT8}, num_output_elements, mask_state::UNALLOCATED, stream, mr);
    auto d_chars = byte_column->mutable_view().data<char>();

    rmm::device_buffer null_mask = copy_bitmask(input_column, stream, mr);
    auto normalized              = normalize_nans_and_zeros(input_column);

    if (configuration == endianess_policy::FLIP_ENDIANESS) {
      uint32_t mask = cudf::size_of(input_column.type()) - 1;
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_output_elements),
                       flip_endianness{d_chars, normalized->view().data<char>(), mask});
    } else {
      thrust::copy_n(rmm::exec_policy(stream)->on(stream),
                     normalized->view().data<char>(),
                     num_output_elements,
                     d_chars);
    }
    return make_lists_column(input_column.size(),
                             std::move(offsets_column),
                             std::move(byte_column),
                             input_column.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
  }

  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     endianess_policy configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    size_type num_output_elements = input_column.size() * cudf::size_of(input_column.type());

    auto begin          = thrust::make_constant_iterator(cudf::size_of(input_column.type()));
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      begin, begin + input_column.size(), mr, stream);

    auto byte_column = make_numeric_column(
      data_type{type_id::UINT8}, num_output_elements, mask_state::UNALLOCATED, stream, mr);
    auto bytes_view = byte_column->mutable_view();
    auto d_chars    = bytes_view.data<char>();
    auto d_data     = input_column.data<char>();

    rmm::device_buffer null_mask = copy_bitmask(input_column, stream, mr);

    if (configuration == endianess_policy::FLIP_ENDIANESS) {
      uint32_t mask = cudf::size_of(input_column.type()) - 1;
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_output_elements),
                       flip_endianness{d_chars, d_data, mask});
    } else {
      thrust::copy_n(rmm::exec_policy(stream)->on(stream), d_data, num_output_elements, d_chars);
    }

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
std::unique_ptr<cudf::column> ByteListConversion::operator()<string_view>(
  column_view const& input_column,
  endianess_policy configuration,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  strings_column_view input_strings(input_column);
  auto strings_count = input_strings.size();
  if (strings_count == 0) return cudf::strings::detail::make_empty_strings_column(mr, stream);

  auto chars_column            = std::make_unique<column>(input_strings.chars(), stream, mr);
  auto offsets_column          = std::make_unique<column>(input_strings.offsets(), stream, mr);
  rmm::device_buffer null_mask = copy_bitmask(input_column, stream, mr);

  return make_lists_column(input_column.size(),
                           std::move(offsets_column),
                           std::move(chars_column),
                           input_column.null_count(),
                           std::move(null_mask),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> byte_cast(column_view const& input_column,
                                  endianess_policy configuration,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    cudf::is_numeric(input_column.type()) || input_column.type().id() == cudf::type_id::STRING,
    "numeric type required to convert to big endian byte list");

  return type_dispatcher(
    input_column.type(), detail::ByteListConversion{}, input_column, configuration, mr, stream);
}

}  // namespace cudf
