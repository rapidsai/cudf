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

#include <thrust/iterator/transform_iterator.h>

#include <iterator>

namespace cudf {
namespace test {
namespace iterators {
/**
 * @brief Bool iterator for marking (possibly multiple) null elements in a column_wrapper.
 *
 * The returned iterator yields `false` (to mark `null`) at all the specified indices,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto indices = std::vector<size_type>{8,9};
 * auto iter = nulls_at(indices.cbegin(), indices.end());
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
[[maybe_unused]] static auto nulls_at(Iter index_start, Iter index_end)
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
 * auto iter = nulls_at({8,9});
 * iter[6] == true;  // i.e. Valid row at index 6.
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * iter[9] == false; // i.e. Invalid row at index 9.
 * @endcode
 *
 * @param indices The indices for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto nulls_at(std::vector<cudf::size_type> const& indices)
{
  return nulls_at(indices.cbegin(), indices.cend());
}

/**
 * @brief Bool iterator for marking a single null element in a column_wrapper
 *
 * The returned iterator yields `false` (to mark `null`) at the specified index,
 * and yields `true` (to mark valid rows) for all other indices. E.g.
 *
 * @code
 * auto iter = null_at(8);
 * iter[7] == true;  // i.e. Valid row at index 7.
 * iter[8] == false; // i.e. Invalid row at index 8.
 * @endcode
 *
 * @param index The index for which the validity iterator must return `false` (i.e. null)
 * @return auto Validity iterator
 */
[[maybe_unused]] static auto null_at(cudf::size_type index)
{
  return nulls_at(std::vector<cudf::size_type>{index});
}

/**
 * @brief Bool iterator for marking all elements are null
 *
 * @return auto Validity iterator which always yields `false`
 */
[[maybe_unused]] static auto all_nulls() { return thrust::make_constant_iterator(false); }

/**
 * @brief Bool iterator for marking all elements are valid (non-null)
 *
 * @return auto Validity iterator which always yields `true`
 */
[[maybe_unused]] static auto no_nulls() { return thrust::make_constant_iterator(true); }

/**
 * @brief Bool iterator for marking null elements from pointers of data
 *
 * The returned iterator yields `false` (to mark `null`) at the indices corresponding to the
 * pointers having `nullptr` values and `true` for the remaining indices.
 *
 * @tparam T the data type
 * @param ptrs The data pointers for which the validity iterator is computed
 * @return auto Validity iterator
 */
template <class T>
[[maybe_unused]] static auto nulls_from_nullptrs(std::vector<T const*> const& ptrs)
{
  // The vector `indices` is copied into the lambda as it can be destroyed at the caller site.
  return thrust::make_transform_iterator(ptrs.begin(), [ptrs](auto ptr) { return ptr != nullptr; });
}

}  // namespace iterators
}  // namespace test
}  // namespace cudf
