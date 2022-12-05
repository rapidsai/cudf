/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(separator.is_valid(stream), "Parameter separator must be a valid string_scalar");

  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // create an offsets array for building the output memory layout
  rmm::device_uvector<size_type> output_offsets(strings_count + 1, stream);
  auto d_output_offsets = output_offsets.data();
  // using inclusive-scan to compute last entry which is the total size
  thrust::transform_inclusive_scan(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    d_output_offsets + 1,
    [d_strings, d_separator, d_narep] __device__(size_type idx) {
      size_type bytes = 0;
      if (d_strings.is_null(idx)) {
        if (!d_narep.is_valid()) return 0;  // skip nulls
        bytes += d_narep.size();
      } else
        bytes += d_strings.element<string_view>(idx).size_bytes();
      if ((idx + 1) < d_strings.size()) bytes += d_separator.size_bytes();
      return bytes;
    },
    thrust::plus<size_type>());

  output_offsets.set_element_to_zero_async(0, stream);
  // total size is the last entry
  size_type const bytes = output_offsets.back_element(stream);

  // build offsets column (only 1 string so 2 offset entries)
  auto offsets_column =
    make_numeric_column(data_type{type_id::INT32}, 2, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  // set the first entry to 0 and the last entry to bytes
  int32_t new_offsets[] = {0, static_cast<int32_t>(bytes)};
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_view.data<int32_t>(),
                                new_offsets,
                                sizeof(new_offsets),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  // build null mask
  // only one entry so it is either all valid or all null
  auto const null_count =
    static_cast<size_type>(strings.null_count() == strings_count && !narep.is_valid(stream));
  auto null_mask    = null_count
                        ? cudf::detail::create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr)
                        : rmm::device_buffer{0, stream, mr};
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [d_strings, d_separator, d_narep, d_output_offsets, d_chars] __device__(size_type idx) {
      size_type offset = d_output_offsets[idx];
      char* d_buffer   = d_chars + offset;
      if (d_strings.is_null(idx)) {
        if (!d_narep.is_valid())
          return;  // do not write to buffer if element is null (including separator)
        d_buffer = detail::copy_string(d_buffer, d_narep.value());
      } else {
        string_view d_str = d_strings.element<string_view>(idx);
        d_buffer          = detail::copy_string(d_buffer, d_str);
      }
      if ((idx + 1) < d_strings.size()) d_buffer = detail::copy_string(d_buffer, d_separator);
    });

  return make_strings_column(
    1, std::move(offsets_column), std::move(chars_column), null_count, std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::join_strings(strings, separator, narep, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
