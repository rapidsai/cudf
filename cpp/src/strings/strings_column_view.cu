/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <iostream>

namespace cudf {
//
strings_column_view::strings_column_view(column_view strings_column) : column_view(strings_column)
{
  CUDF_EXPECTS(type().id() == type_id::STRING, "strings_column_view only supports strings");
}

column_view strings_column_view::parent() const { return static_cast<column_view>(*this); }

column_view strings_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() > 0, "strings column has no children");
  return child(offsets_column_index);
}

column_view strings_column_view::chars() const
{
  CUDF_EXPECTS(num_children() > 0, "strings column has no children");
  return child(chars_column_index);
}

size_type strings_column_view::chars_size() const noexcept
{
  if (size() == 0) return 0;
  return chars().size();
}

namespace strings {
// print strings to stdout
void print(strings_column_view const& strings,
           size_type first,
           size_type last,
           size_type max_width,
           const char* delimiter)
{
  size_type count = strings.size();
  if (last < 0 || last > count) last = count;
  if (first < 0) first = 0;
  CUDF_EXPECTS(((first >= 0) && (first < last)), "invalid start parameter");
  count = last - first;

  // stick with the default stream for this odd/rare stdout function
  auto strings_column = column_device_view::create(strings.parent());
  auto d_column       = *strings_column;

  // create output strings offsets
  rmm::device_vector<size_type> output_offsets(count + 1);
  size_type* d_output_offsets = output_offsets.data().get();
  thrust::transform_inclusive_scan(
    thrust::device,
    thrust::make_counting_iterator<size_type>(first),
    thrust::make_counting_iterator<size_type>(last),
    d_output_offsets + 1,
    [d_column, max_width] __device__(size_type idx) {
      if (d_column.is_null(idx)) return static_cast<size_type>(0);
      string_view d_str = d_column.element<string_view>(idx);
      size_type bytes   = d_str.size_bytes();
      if ((max_width > 0) && (d_str.length() > max_width)) bytes = d_str.byte_offset(max_width);
      return static_cast<size_type>(bytes + 1);  // allow for null-terminator on non-null strings
    },
    thrust::plus<size_type>());
  CUDA_TRY(cudaMemset(d_output_offsets, 0, sizeof(*d_output_offsets)));
  // build output buffer
  size_type buffer_size = output_offsets.back();  // last element has total size
  if (buffer_size == 0) {
    std::cout << "all " << count << " strings are null\n";
    return;
  }
  rmm::device_vector<char> buffer(buffer_size, 0);  // allocate and pre-null-terminate
  char* d_buffer = buffer.data().get();
  // copy strings into output buffer
  thrust::for_each_n(
    thrust::device,
    thrust::make_counting_iterator<size_type>(0),
    count,
    [d_column, max_width, first, d_output_offsets, d_buffer] __device__(size_type idx) {
      if (d_column.is_null(first + idx)) return;
      string_view d_str = d_column.element<string_view>(first + idx);
      size_type bytes   = d_str.size_bytes();
      if ((max_width > 0) && (d_str.length() > max_width)) bytes = d_str.byte_offset(max_width);
      memcpy(d_buffer + d_output_offsets[idx], d_str.data(), bytes);
    });

  // copy output buffer to host
  std::vector<size_type> h_offsets(count + 1);
  CUDA_TRY(cudaMemcpy(
    h_offsets.data(), d_output_offsets, (count + 1) * sizeof(size_type), cudaMemcpyDeviceToHost));
  std::vector<char> h_buffer(buffer_size);
  CUDA_TRY(cudaMemcpy(h_buffer.data(), d_buffer, buffer_size, cudaMemcpyDeviceToHost));

  // print out the strings to stdout
  for (size_type idx = 0; idx < count; ++idx) {
    size_type offset = h_offsets[idx];
    size_type length = h_offsets[idx + 1] - offset;
    std::cout << idx << ":";
    if (length)
      std::cout << "[" << std::string(h_buffer.data() + offset) << "]";
    else
      std::cout << "<null>";
    std::cout << delimiter;
  }
}

//
std::pair<rmm::device_vector<char>, rmm::device_vector<size_type>> create_offsets(
  strings_column_view const& strings, cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  size_type count          = strings.size();
  const int32_t* d_offsets = strings.offsets().data<int32_t>();
  d_offsets += strings.offset();  // nvbug-2808421 : do not combine with the previous line
  int32_t first = 0;
  CUDA_TRY(cudaMemcpyAsync(&first, d_offsets, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  rmm::device_vector<size_type> offsets(count + 1);
  // normalize the offset values for the column offset
  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    d_offsets,
    d_offsets + count + 1,
    offsets.begin(),
    [first] __device__(int32_t offset) { return static_cast<size_type>(offset - first); });
  // copy the chars column data
  int32_t bytes = 0;  // last offset entry is the size in bytes
  CUDA_TRY(
    cudaMemcpyAsync(&bytes, d_offsets + count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  bytes -= first;
  const char* d_chars = strings.chars().data<char>() + first;
  rmm::device_vector<char> chars(bytes);
  CUDA_TRY(cudaMemcpyAsync(chars.data().get(), d_chars, bytes, cudaMemcpyDeviceToHost, stream));
  // return offsets and chars
  return std::make_pair(std::move(chars), std::move(offsets));
}

}  // namespace strings
}  // namespace cudf
