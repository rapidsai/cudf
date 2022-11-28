/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf {
namespace detail {

/**
 * @brief Iterator that can be used with a scan algorithm to also perform reduce
 *
 * Use cudf::detail::make_scan_reduce_output_iterator to create an instance of this class.
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam ReduceType Type used for final scan result
 */
template <typename ScanIterator, typename ReduceType>
struct scan_reduce_iterator {
  using difference_type   = ptrdiff_t;
  using value_type        = ReduceType;
  using pointer           = ReduceType*;
  using reference         = scan_reduce_iterator const&;
  using iterator_category = std::random_access_iterator_tag;

  using ScanType = typename thrust::iterator_traits<ScanIterator>::value_type;

  CUDF_HOST_DEVICE inline scan_reduce_iterator& operator++()
  {
    ++itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator operator++(int)
  {
    scan_reduce_iterator tmp(*this);
    operator++();
    return tmp;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator& operator--()
  {
    --itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator operator--(int)
  {
    scan_reduce_iterator tmp(*this);
    operator--();
    return tmp;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator& operator+=(difference_type offset)
  {
    itr_ += offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator operator+(difference_type offset) const
  {
    scan_reduce_iterator tmp(*this);
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend scan_reduce_iterator operator+(difference_type offset,
                                                                scan_reduce_iterator const& rhs)
  {
    scan_reduce_iterator tmp{rhs};
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator& operator-=(difference_type offset)
  {
    itr_ -= offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator operator-(difference_type offset) const
  {
    scan_reduce_iterator tmp(*this);
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend scan_reduce_iterator operator-(difference_type offset,
                                                                scan_reduce_iterator const& rhs)
  {
    scan_reduce_iterator tmp{rhs};
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline difference_type operator-(scan_reduce_iterator const& rhs) const
  {
    return itr_ - rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator==(scan_reduce_iterator const& rhs) const
  {
    return rhs.itr_ == itr_;
  }
  CUDF_HOST_DEVICE inline bool operator!=(scan_reduce_iterator const& rhs) const
  {
    return rhs.itr_ != itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<(scan_reduce_iterator const& rhs) const
  {
    return itr_ < rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>(scan_reduce_iterator const& rhs) const
  {
    return itr_ > rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<=(scan_reduce_iterator const& rhs) const
  {
    return itr_ <= rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>=(scan_reduce_iterator const& rhs) const
  {
    return itr_ >= rhs.itr_;
  }

  CUDF_HOST_DEVICE inline scan_reduce_iterator const& operator*() const { return *this; }

  CUDF_HOST_DEVICE inline scan_reduce_iterator const operator[](int idx) const
  {
    scan_reduce_iterator tmp{*this};
    tmp.itr_ += idx;
    return tmp;
  }

  /**
   * @brief Called to set the output of the scan operation to the current iterator position
   *
   * @param value Value to set to the current output
   * @return This iterator instance
   */
  CUDF_HOST_DEVICE inline scan_reduce_iterator const& operator=(ReduceType const value) const
  {
    *itr_ = static_cast<ScanType>(value);  // place into the output
    if (itr_ == end_) { *last_ = value; }  // also save the last value
    return *this;
  }

  scan_reduce_iterator()                            = default;
  scan_reduce_iterator(scan_reduce_iterator const&) = default;
  scan_reduce_iterator(scan_reduce_iterator&&)      = default;
  scan_reduce_iterator& operator=(scan_reduce_iterator const&) = default;
  scan_reduce_iterator& operator=(scan_reduce_iterator&&) = default;

 protected:
  template <typename S, typename R>
  friend scan_reduce_iterator<S, R> make_scan_reduce_output_iterator(S, S, R*);

  /**
   * @brief Iterator constructor
   *
   * Use the make_scan_reduce_output_iterator() to create an instance of this class
   */
  scan_reduce_iterator(ScanIterator begin, ScanIterator end, ReduceType* last)
    : itr_{begin}, end_{thrust::prev(end)}, last_{last}
  {
  }

  ScanIterator itr_{};
  ScanIterator end_{};
  ReduceType* last_{};
};

/**
 * @brief Create an instance of a scan_reduce_iterator
 *
 * @code{.pseudo}
 *  auto begin = // begin input iterator
 *  auto end = // end input iterator
 *  auto result = rmm::device_uvector(std::distance(begin,end), stream);
 *  auto reduction = rmm::device_scalar<int64_t>(0, stream);
 *  auto itr = make_scan_reduce_output_iterator(result.begin(),
 *                                              result.end(),
 *                                              reduction.data());
 *  thrust::exclusive_scan(rmm::exec_policy(stream), begin, end, itr, int64_t{0});
 *  // reduction contains the reduce result
 * @endcode
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam ReduceType Type used for accumulating the reduce operation
 *
 * @param begin Output iterator for scan
 * @param end End of the output iterator for scan
 * @param reduction Reduce operation result is stored here
 * @return Instance of iterator
 */
template <typename ScanIterator, typename ReduceType>
static scan_reduce_iterator<ScanIterator, ReduceType> make_scan_reduce_output_iterator(
  ScanIterator begin, ScanIterator end, ReduceType* reduction)
{
  return scan_reduce_iterator<ScanIterator, ReduceType>(begin, end, reduction);
}

/**
 * @brief Perform a combination exclusive-scan and reduce on the given input
 *
 * This performs an exclusive-scan and reduce (addition only) on the given input `[begin, end)`.
 * The output of the scan is placed in `result` and the reduction result is returned.
 *
 * This implementation will return the reduction result in `int64_t` or `uint64_t` precision
 * as appropriate regardless of the input or result types.
 * This can be used to check if the scan will overflow when the input and result are declared
 * as smaller types (i.e. computing offsets from sizes).
 *
 * Only integral types for input and result types are supported.
 *
 * Note that `begin == result` is allowed but `result` may not overlap `[begin,end)` otherwise the
 * behavior is undefined.
 *
 * @code{.pseudo}
 *   auto const bytes = cudf::detail::exclusive_scan_reduce(
 *     d_offsets, d_offsets + strings_count + 1, d_offsets, stream);
 *   CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
 *               "Size of output exceeds column size limit");
 * @endcode
 *
 * @tparam ScanIterator Iterator type for input and output of the scan using addition operation
 *
 * @param begin Input iterator for scan/reduce
 * @param end End of the input iterator
 * @param result Output iterator for scan result
 * @return Result of the reduce operation
 */
template <typename ScanIterator>
auto exclusive_scan_reduce(ScanIterator begin,
                           ScanIterator end,
                           ScanIterator result,
                           rmm::cuda_stream_view stream)
{
  using ScanType = typename thrust::iterator_traits<ScanIterator>::value_type;
  static_assert(std::is_integral_v<ScanType>,
                "Only numeric types are supported by exclusive_scan_reduce");

  using ReduceType = std::conditional_t<std::is_signed_v<ScanType>, int64_t, uint64_t>;
  auto reduction   = rmm::device_scalar<ReduceType>(0, stream);
  auto output_itr =
    make_scan_reduce_output_iterator(result, result + std::distance(begin, end), reduction.data());
  // This function uses the type of the initialization parameter as the accumulator type
  // when computing the individual scan output elements.
  thrust::exclusive_scan(rmm::exec_policy(stream), begin, end, output_itr, ReduceType{0});
  return reduction.value(stream);
}

}  // namespace detail
}  // namespace cudf
