/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/tdigest/tdigest_column_view.hpp>

namespace cudf {
namespace tdigest {
namespace detail {

/**
 * @brief Functor to compute the size of each tdigest of a column
 */
struct tdigest_size_fn {
  size_type const* offsets;  ///< Offsets of the t-digest column
  /**
   * @brief Returns size of the each tdigest in the column
   *
   * @param tdigest_index Index of the tdigest in the column
   * @return Size of the tdigest
   */
  __device__ size_type operator()(size_type tdigest_index)
  {
    return offsets[tdigest_index + 1] - offsets[tdigest_index];
  }
};

/**
 * @brief Returns an iterator that returns the size of each tdigest
 * in the column (each row is 1 digest)
 *
 * @return An iterator that returns the size of each tdigest in the column
 */
inline auto size_begin(tdigest_column_view const& tdv)
{
  return cudf::detail::make_counting_transform_iterator(
    0, tdigest_size_fn{tdv.centroids().offsets_begin()});
}

}  // namespace detail
}  // namespace tdigest
}  // namespace cudf
