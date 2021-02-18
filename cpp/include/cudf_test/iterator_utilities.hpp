/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <iterator>

namespace cudf {
namespace test {

/**
 * @brief Bool iterator for marking (possibly multiple) null elements in a column_wrapper.
 *
 * The returned iterator yields `false` (to mark `null`) at all the specified indices,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto indices = std::vector<size_type>{8,9};
 * auto iter = iterator_with_null_at(indices.cbegin(), indices.end());
 * iter[6] == true;  // i.e. Valid row at index 6.
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * iter[9] == false; // i.e. Invalid row at index 9.
 * @endcode
 *
 * @tparam Iter Iterator type
 * @param index_start Iterator to start of indices for which the validity iterator
 *                    must return `false` (i.e. null)
 * @param index_end   Iterator to end of indices for the validity iterator
 * @return auto Validity iterator
 */
template <typename Iter>
static auto iterator_with_null_at(Iter index_start, Iter index_end)
{
  using index_type = typename std::iterator_traits<Iter>::value_type;

  return cudf::detail::make_counting_transform_iterator(
    0, [indices = std::vector<index_type>{index_start, index_end}](auto i) {
      return std::find(indices.cbegin(), indices.cend(), i) == indices.cend();
    });
}

/**
 * @brief Bool iterator for marking (possibly multiple) null elements in a column_wrapper.
 *
 * The returned iterator yields `false` (to mark `null`) at all the specified indices,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * using host_span = cudf::detail::host_span<cudf::size_type const>;
 * auto iter = iterator_with_null_at(host_span{std::vector<size_type>{8,9}});
 * iter[6] == true;  // i.e. Valid row at index 6.
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * iter[9] == false; // i.e. Invalid row at index 9.
 * @endcode
 *
 * @param indices The indices for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
static auto iterator_with_null_at(cudf::detail::host_span<cudf::size_type const> const& indices)
{
  return iterator_with_null_at(indices.begin(), indices.end());
}

/**
 * @brief Bool iterator for marking a single null element in a column_wrapper
 *
 * The returned iterator yields `false` (to mark `null`) at the specified index,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto iter = iterator_with_null_at(8);
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * @endcode
 *
 * @param index The index for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
static auto iterator_with_null_at(cudf::size_type const& index)
{
  return iterator_with_null_at(std::vector<size_type>{index});
}

}  // namespace test
}  // namespace cudf
