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

#include "column_buffer.hpp"

#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::io::detail {

template <typename SizesIterator, typename OffsetsIterator>
auto sizes_to_offsets(SizesIterator begin,
                      SizesIterator end,
                      OffsetsIterator result,
                      int64_t str_offset,
                      rmm::cuda_stream_view stream)
{
  using SizeType = typename thrust::iterator_traits<SizesIterator>::value_type;
  static_assert(std::is_integral_v<SizeType>,
                "Only numeric types are supported by sizes_to_offsets");

  using LastType    = std::conditional_t<std::is_signed_v<SizeType>, int64_t, uint64_t>;
  auto last_element = rmm::device_scalar<LastType>(0, stream);
  auto output_itr   = cudf::detail::make_sizes_to_offsets_iterator(
    result, result + std::distance(begin, end), last_element.data());
  // This function uses the type of the initialization parameter as the accumulator type
  // when computing the individual scan output elements.
  thrust::exclusive_scan(rmm::exec_policy(stream), begin, end, output_itr, LastType{str_offset});
  return last_element.value(stream);
}

std::unique_ptr<column> cudf::io::detail::inline_column_buffer::make_string_column_impl(
  rmm::cuda_stream_view stream)
{
  // if the size of _string_data is over the threshold for 64bit size_type, _data will contain
  // sizes rather than offsets. need special handling for that case.
  if (is_large_strings_column()) {
    if (not strings::detail::is_large_strings_enabled()) {
      CUDF_FAIL("String column exceeds the column size limit", std::overflow_error);
    }
    // create new offsets
    auto const offsets_ptr = static_cast<size_type*>(_data.data());
    auto offsets_col       = make_numeric_column(
      data_type{type_id::INT64}, size + 1, mask_state::UNALLOCATED, stream, _mr);
    auto d_offsets64 = offsets_col->mutable_view().template data<int64_t>();
    // std::cout << "Offset64 = " << str_offset << std::endl;
    //   it's safe to call with size + 1 because _data is also sized that large
    sizes_to_offsets(offsets_ptr, offsets_ptr + size + 1, d_offsets64, str_offset, stream);
    // auto h_offsets64 = cudf::detail::make_host_vector_sync(
    //   device_span<int64_t const>{d_offsets64, static_cast<size_t>(size + 1)}, stream);
    // std::cout << "Offset64[0] = " << h_offsets64[0] << std::endl;
    return make_strings_column(
      size, std::move(offsets_col), std::move(_string_data), null_count(), std::move(_null_mask));
  } else {
    // no need for copies, just transfer ownership of the data_buffers to the columns
    auto offsets_col = std::make_unique<column>(
      data_type{type_to_id<size_type>()}, size + 1, std::move(_data), rmm::device_buffer{}, 0);
    // auto d_offsets32 = offsets_col->mutable_view().template data<size_type>();
    // auto h_offsets32 = cudf::detail::make_host_vector_sync(
    //   device_span<size_type const>{d_offsets32, static_cast<size_t>(size + 1)}, stream);
    // std::cout << "Offset32[0] = " << h_offsets32[0] << std::endl;
    return make_strings_column(
      size, std::move(offsets_col), std::move(_string_data), null_count(), std::move(_null_mask));
  }
}

}  // namespace cudf::io::detail
