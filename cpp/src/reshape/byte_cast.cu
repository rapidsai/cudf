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
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace {

struct byte_list_conversion {
  /**
   * @brief Function object for converting primitive types and string columns to lists of bytes,
   * mimics Spark's cast to binary type.
   */
  struct flip_endianness_lambda {
    char* d_chars;
    const char* d_data;
    uint32_t mask;
    __device__ void operator()(int byte_index)
    {
      d_chars[byte_index] = d_data[byte_index + mask - ((byte_index & mask) << 1)];
    }
  };

  template <typename T>
  struct normalize_lambda {
    T* d_result;
    const T* d_data;
    __device__ void operator()(int index)
    {
      auto e = d_data[index];
      if (isnan(e)) {
        d_result[index] = std::numeric_limits<T>::quiet_NaN();
      } else if (T{0.0} == e) {
        d_result[index] = T{0.0};
      } else
        d_result[index] = e;
    }
  };

  template <typename T>
  struct normalize_flip_endianess_lambda {
    char* d_chars;
    const T* d_data;
    size_t size;
    __device__ void operator()(int index)
    {
      T normal           = normalize(d_data[index]);
      char* normal_bytes = reinterpret_cast<char*>(&normal);
      for (int i = 0; i < size; i++) { d_chars[index * size + i] = normal_bytes[size - 1 - i]; }
    }

    __device__ T normalize(T in)
    {
      if (isnan(in)) { return std::numeric_limits<T>::quiet_NaN(); }
      if (T{0.0} == in) { return T{0.0}; }
      return in;
    }
  };

  template <typename T,
            std::enable_if_t<!std::is_integral<T>::value and !is_floating_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     flip_endianness configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    CUDF_FAIL("Unsupported non-numeric and non-string column");
  }

  template <typename T, std::enable_if_t<is_floating_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     flip_endianness configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    auto byte_column = make_numeric_column(data_type{type_id::UINT8},
                                           input_column.size() * cudf::size_of(input_column.type()),
                                           mask_state::UNALLOCATED,
                                           stream,
                                           mr);

    if (configuration == flip_endianness::YES) {
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(input_column.size()),
                       normalize_flip_endianess_lambda<T>{
                         reinterpret_cast<char*>(byte_column->mutable_view().data<T>()),
                         input_column.data<T>(),
                         cudf::size_of(input_column.type())});
    } else {
      thrust::for_each(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(input_column.size()),
        normalize_lambda<T>{byte_column->mutable_view().data<T>(), input_column.data<T>()});
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

  template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input_column,
                                     flip_endianness configuration,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    size_type num_output_elements = input_column.size() * cudf::size_of(input_column.type());

    auto byte_column = make_numeric_column(
      data_type{type_id::UINT8}, num_output_elements, mask_state::UNALLOCATED, stream, mr);
    auto d_chars = reinterpret_cast<char*>(byte_column->mutable_view().data<T>());

    if (configuration == flip_endianness::YES) {
      uint32_t mask = cudf::size_of(input_column.type()) - 1;
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_output_elements),
                       flip_endianness_lambda{
                         d_chars, reinterpret_cast<const char*>(input_column.data<T>()), mask});
    } else {
      thrust::copy_n(rmm::exec_policy(stream)->on(stream),
                     reinterpret_cast<const char*>(input_column.data<T>()),
                     num_output_elements,
                     d_chars);
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
