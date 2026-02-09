/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

/**
￼ * @brief Device functor to determine if a row is valid.
￼ */
class row_validity {
 public:
  row_validity(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ inline bool operator()(size_type const& i) const noexcept
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
  size_type const last_index = cuda::std::distance(first, last) - 1;
  return cudf::detail::copy_if(
    first,
    last,
    thrust::counting_iterator<size_type>(0),
    output,
    unique_copy_fn<InputIterator, BinaryPredicate>{first, keep, comp, last_index},
    stream);
}
}  // namespace detail
}  // namespace cudf
