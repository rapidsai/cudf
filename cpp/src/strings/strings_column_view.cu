/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

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

std::pair<rmm::device_uvector<char>, rmm::device_uvector<size_type>> create_offsets(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  size_type const count = strings.size();

  auto d_offsets = strings.offsets().data<int32_t>();
  d_offsets += strings.offset();  // nvbug-2808421 : do not combine with the previous line

  rmm::device_uvector<size_type> offsets(count + 1, stream);
  // normalize the offset values for the column offset
  thrust::transform(rmm::exec_policy(stream),
                    d_offsets,
                    d_offsets + count + 1,
                    offsets.begin(),
                    [d_offsets] __device__(int32_t offset) {
                      return static_cast<size_type>(offset - d_offsets[0]);
                    });

  // get the input chars column byte offset
  auto const bytes = offsets.element(count, stream);
  auto const chars_offset =
    cudf::detail::get_value<offset_type>(strings.offsets(), strings.offset(), stream);
  stream.synchronize();

  // copy the chars column data
  const char* d_chars = strings.chars().data<char>() + chars_offset;
  rmm::device_uvector<char> chars(bytes, stream);
  CUDA_TRY(cudaMemcpyAsync(chars.data(), d_chars, bytes, cudaMemcpyDefault, stream.value()));

  // return offsets and chars
  return std::make_pair(std::move(chars), std::move(offsets));
}

}  // namespace strings
}  // namespace cudf
