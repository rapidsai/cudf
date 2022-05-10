/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "stream_compaction_common.hpp"

#include <cudf/stream_compaction.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

/**
 * @brief Device callable to hash a given row.
 */
template <typename Nullate>
class compaction_hash {
 public:
  compaction_hash(Nullate has_nulls, table_device_view t) : _hash{has_nulls, t} {}

  __device__ inline auto operator()(size_type i) const noexcept
  {
    auto hash = _hash(i);
    return (hash == COMPACTION_EMPTY_KEY_SENTINEL) ? (hash - 1) : hash;
  }

 private:
  row_hash _hash;
};

namespace experimental {

/**
 * @brief Device callable to hash a given row.
 */
template <typename RowHash>
class compaction_hash {
 public:
  compaction_hash(RowHash row_hasher) : _hash{row_hasher} {}

  __device__ inline auto operator()(size_type i) const noexcept
  {
    auto hash = _hash(i);
    return (hash == COMPACTION_EMPTY_KEY_SENTINEL) ? (hash - 1) : hash;
  }

 private:
  RowHash _hash;
};

}  // namespace experimental

/**
￼ * @brief Device functor to determine if a row is valid.
￼ */
class row_validity {
 public:
  row_validity(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ inline bool operator()(const size_type& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

template <typename InputIterator, typename BinaryPredicate>
struct unique_copy_fn {
  /**
   * @brief Functor for unique_copy()
   *
   * The logic here is equivalent to:
   * @code
   *   ((keep == duplicate_keep_option::KEEP_LAST) ||
   *    (i == 0 || !comp(iter[i], iter[i - 1]))) &&
   *   ((keep == duplicate_keep_option::KEEP_FIRST) ||
   *    (i == last_index || !comp(iter[i], iter[i + 1])))
   * @endcode
   *
   * It is written this way so that the `comp` comparator
   * function appears only once minimizing the inlining
   * required and reducing the compile time.
   */
  __device__ bool operator()(size_type i)
  {
    size_type boundary = 0;
    size_type offset   = 1;
    auto keep_option   = duplicate_keep_option::KEEP_LAST;
    do {
      if ((keep != keep_option) && (i != boundary) && comp(iter[i], iter[i - offset])) {
        return false;
      }
      keep_option = duplicate_keep_option::KEEP_FIRST;
      boundary    = last_index;
      offset      = -offset;
    } while (offset < 0);
    return true;
  }

  InputIterator iter;
  duplicate_keep_option const keep;
  BinaryPredicate comp;
  size_type const last_index;
};

/**
 * @brief Copies unique elements from the range [first, last) to output iterator `output`.
 *
 * In a consecutive group of duplicate elements, depending on parameter `keep`,
 * only the first element is copied, or the last element is copied or neither is copied.
 *
 * @return End of the range to which the elements are copied.
 */
template <typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate comp,
                           duplicate_keep_option const keep,
                           rmm::cuda_stream_view stream)
{
  size_type const last_index = thrust::distance(first, last) - 1;
  return thrust::copy_if(
    rmm::exec_policy(stream),
    first,
    last,
    thrust::counting_iterator<size_type>(0),
    output,
    unique_copy_fn<InputIterator, BinaryPredicate>{first, keep, comp, last_index});
}
}  // namespace detail
}  // namespace cudf
