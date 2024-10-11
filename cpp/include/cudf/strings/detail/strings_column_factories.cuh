/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Basic type expected for iterators passed to `make_strings_column` that represent string
 * data in device memory.
 */
using string_index_pair = thrust::pair<char const*, size_type>;

/**
 * @brief Gather characters to create a strings column using the given string_index_pair iterator
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param offsets The offsets for the output strings column
 * @param chars_size The size (in bytes) of the chars data
 * @param begin Iterator to the first string_index_pair
 * @param strings_count The number of strings
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return An array of chars gathered from the input string_index_pair iterator
 */
template <typename IndexPairIterator>
rmm::device_uvector<char> make_chars_buffer(column_view const& offsets,
                                            int64_t chars_size,
                                            IndexPairIterator begin,
                                            size_type strings_count,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto chars_data      = rmm::device_uvector<char>(chars_size, stream, mr);
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets);

  auto const src_ptrs = cudf::detail::make_counting_transform_iterator(
    0u, cuda::proclaim_return_type<void*>([begin] __device__(uint32_t idx) {
      // Due to a bug in cub (https://github.com/NVIDIA/cccl/issues/586),
      // we have to use `const_cast` to remove `const` qualifier from the source pointer.
      // This should be fine as long as we only read but not write anything to the source.
      return reinterpret_cast<void*>(const_cast<char*>(begin[idx].first));
    }));
  auto const src_sizes = cudf::detail::make_counting_transform_iterator(
    0u, cuda::proclaim_return_type<size_type>([begin] __device__(uint32_t idx) {
      return begin[idx].second;
    }));
  auto const dst_ptrs = cudf::detail::make_counting_transform_iterator(
    0u,
    cuda::proclaim_return_type<char*>([offsets = d_offsets, output = chars_data.data()] __device__(
                                        uint32_t idx) { return output + offsets[idx]; }));

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, src_ptrs, dst_ptrs, src_sizes, strings_count, stream.value()));
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(d_temp_storage.data(),
                                           temp_storage_bytes,
                                           src_ptrs,
                                           dst_ptrs,
                                           src_sizes,
                                           strings_count,
                                           stream.value()));

  return chars_data;
}

/**
 * @brief Create a strings-type column from iterators of pointer/size pairs
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param begin First string row (inclusive)
 * @param end Last string row (exclusive)
 * @param stream CUDA stream used for device memory operations
 * @param mr  Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename IndexPairIterator>
std::unique_ptr<column> make_strings_column(IndexPairIterator begin,
                                            IndexPairIterator end,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = thrust::distance(begin, end);
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column from the strings sizes
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([] __device__(string_index_pair item) -> size_type {
      return (item.first != nullptr ? static_cast<size_type>(item.second) : size_type{0});
    });
  auto offsets_transformer_itr = thrust::make_transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  auto new_nulls = cudf::detail::valid_if(begin, end, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask =
    (null_count > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};

  // build chars column
  auto chars_data =
    make_chars_buffer(offsets_column->view(), bytes, begin, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

/**
 * @brief Create a strings-type column from iterators to chars, offsets, and bitmask.
 *
 * @tparam CharIterator iterator over character bytes (int8)
 * @tparam OffsetIterator iterator over offset values (size_type)
 *
 * @param chars_begin First character byte (inclusive)
 * @param chars_end Last character byte (exclusive)
 * @param offset_begin First offset value (inclusive)
 * @param offset_end Last offset value (exclusive)
 * @param null_count Number of null rows
 * @param null_mask The validity bitmask in Arrow format
 * @param stream CUDA stream used for device memory operations
 * @param mr  Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename CharIterator, typename OffsetIterator>
std::unique_ptr<column> make_strings_column(CharIterator chars_begin,
                                            CharIterator chars_end,
                                            OffsetIterator offsets_begin,
                                            OffsetIterator offsets_end,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = thrust::distance(offsets_begin, offsets_end) - 1;
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  int64_t const bytes = std::distance(chars_begin, chars_end) * sizeof(char);
  CUDF_EXPECTS(bytes >= 0, "invalid offsets data");

  // build offsets column -- this is the number of strings + 1
  auto [offsets_column, computed_bytes] =
    cudf::strings::detail::make_offsets_child_column(offsets_begin, offsets_end, stream, mr);
  CUDF_EXPECTS(bytes == computed_bytes, "unexpected byte count");

  // build chars column
  rmm::device_uvector<char> chars_data(bytes, stream, mr);
  thrust::copy(rmm::exec_policy(stream), chars_begin, chars_end, chars_data.begin());

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
