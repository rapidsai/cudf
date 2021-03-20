/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <strings/utilities.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Average string byte-length threshold for deciding character-level
 * vs. row-level parallel algorithm.
 *
 * This value was determined by running the factory_benchmark against different
 * string lengths and observing the point where the performance is faster for
 * long strings.
 */
constexpr size_type FACTORY_BYTES_PER_ROW_THRESHOLD = 64;

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
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = thrust::distance(begin, end);
  if (strings_count == 0) return strings::detail::make_empty_strings_column(stream, mr);

  using string_index_pair = thrust::pair<const char*, size_type>;

  // check total size is not too large for cudf column
  auto size_checker = [] __device__(string_index_pair const& item) {
    return (item.first != nullptr) ? item.second : 0;
  };
  size_t const bytes = thrust::transform_reduce(
    rmm::exec_policy(stream), begin, end, size_checker, 0, thrust::plus<size_t>());
  CUDF_EXPECTS(bytes < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "total size of strings is too large for cudf column");

  // build offsets column from the strings sizes
  auto offsets_transformer = [] __device__(string_index_pair item) {
    return (item.first != nullptr ? static_cast<int32_t>(item.second) : 0);
  };
  auto offsets_transformer_itr = thrust::make_transform_iterator(begin, offsets_transformer);
  auto offsets_column          = strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  auto new_nulls = cudf::detail::valid_if(begin, end, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask =
    (null_count > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};

  auto const avg_bytes_per_row = bytes / std::max(strings_count - null_count, 1);
  // build chars column
  std::unique_ptr<column> chars_column = [&] {
    // use a character-parallel kernel for long string lengths
    if (avg_bytes_per_row > FACTORY_BYTES_PER_ROW_THRESHOLD) {
      auto const d_offsets =
        device_span<size_type const>{offsets_column->view().template data<int32_t>(),
                                     static_cast<std::size_t>(offsets_column->size())};
      auto const str_begin = thrust::make_transform_iterator(begin, [] __device__(auto ip) {
        return string_view{ip.first, ip.second};
      });

      return gather_chars(str_begin,
                          thrust::make_counting_iterator<size_type>(0),
                          thrust::make_counting_iterator<size_type>(strings_count),
                          d_offsets,
                          static_cast<size_type>(bytes),
                          stream,
                          mr);
    } else {
      // this approach is 2-3x faster for a large number of smaller string lengths
      auto chars_column =
        strings::detail::create_chars_child_column(strings_count, null_count, bytes, stream, mr);
      auto d_chars    = chars_column->mutable_view().template data<char>();
      auto copy_chars = [d_chars] __device__(auto item) {
        string_index_pair const str = thrust::get<0>(item);
        size_type const offset      = thrust::get<1>(item);
        if (str.first != nullptr) memcpy(d_chars + offset, str.first, str.second);
      };
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::make_zip_iterator(thrust::make_tuple(
                           begin, offsets_column->view().template begin<int32_t>())),
                         strings_count,
                         copy_chars);
      return chars_column;
    }
  }();

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
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
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = thrust::distance(offsets_begin, offsets_end) - 1;
  size_type bytes         = std::distance(chars_begin, chars_end) * sizeof(char);
  if (strings_count == 0) return strings::detail::make_empty_strings_column(stream, mr);

  CUDF_EXPECTS(null_count < strings_count, "null strings column not yet supported");
  CUDF_EXPECTS(bytes >= 0, "invalid offsets data");

  // build offsets column -- this is the number of strings + 1
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  thrust::transform(rmm::exec_policy(stream),
                    offsets_begin,
                    offsets_end,
                    offsets_view.data<int32_t>(),
                    [] __device__(auto offset) { return static_cast<int32_t>(offset); });

  // build chars column
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, stream, mr);
  auto chars_view = chars_column->mutable_view();
  thrust::copy(rmm::exec_policy(stream), chars_begin, chars_end, chars_view.data<char>());

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
