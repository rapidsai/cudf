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
 * @brief Iterator that can be used with a scan algorithm and also return the last element
 *
 * Use cudf::detail::make_sizes_to_offsets_iterator to create an instance of this class.
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam LastType Type used for final scan element
 */
template <typename ScanIterator, typename LastType>
struct sizes_to_offsets_iterator {
  using difference_type   = ptrdiff_t;
  using value_type        = LastType;
  using pointer           = LastType*;
  using reference         = sizes_to_offsets_iterator const&;
  using iterator_category = std::random_access_iterator_tag;

  using ScanType = typename thrust::iterator_traits<ScanIterator>::value_type;

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator++()
  {
    ++itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator++(int)
  {
    sizes_to_offsets_iterator tmp(*this);
    operator++();
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator--()
  {
    --itr_;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator--(int)
  {
    sizes_to_offsets_iterator tmp(*this);
    operator--();
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator+=(difference_type offset)
  {
    itr_ += offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator+(difference_type offset) const
  {
    sizes_to_offsets_iterator tmp(*this);
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend sizes_to_offsets_iterator operator+(
    difference_type offset, sizes_to_offsets_iterator const& rhs)
  {
    sizes_to_offsets_iterator tmp{rhs};
    tmp.itr_ += offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator& operator-=(difference_type offset)
  {
    itr_ -= offset;
    return *this;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator operator-(difference_type offset) const
  {
    sizes_to_offsets_iterator tmp(*this);
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline friend sizes_to_offsets_iterator operator-(
    difference_type offset, sizes_to_offsets_iterator const& rhs)
  {
    sizes_to_offsets_iterator tmp{rhs};
    tmp.itr_ -= offset;
    return tmp;
  }

  CUDF_HOST_DEVICE inline difference_type operator-(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ - rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator==(sizes_to_offsets_iterator const& rhs) const
  {
    return rhs.itr_ == itr_;
  }
  CUDF_HOST_DEVICE inline bool operator!=(sizes_to_offsets_iterator const& rhs) const
  {
    return rhs.itr_ != itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ < rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ > rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator<=(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ <= rhs.itr_;
  }
  CUDF_HOST_DEVICE inline bool operator>=(sizes_to_offsets_iterator const& rhs) const
  {
    return itr_ >= rhs.itr_;
  }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const& operator*() const { return *this; }

  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const operator[](int idx) const
  {
    sizes_to_offsets_iterator tmp{*this};
    tmp.itr_ += idx;
    return tmp;
  }

  /**
   * @brief Called to set the output of the scan operation to the current iterator position
   *
   * @param value Value to set to the current output
   * @return This iterator instance
   */
  CUDF_HOST_DEVICE inline sizes_to_offsets_iterator const& operator=(LastType const value) const
  {
    *itr_ = static_cast<ScanType>(value);  // place into the output
    if (itr_ == end_) { *last_ = value; }  // also save the last value
    return *this;
  }

  sizes_to_offsets_iterator()                                 = default;
  sizes_to_offsets_iterator(sizes_to_offsets_iterator const&) = default;
  sizes_to_offsets_iterator(sizes_to_offsets_iterator&&)      = default;
  sizes_to_offsets_iterator& operator=(sizes_to_offsets_iterator const&) = default;
  sizes_to_offsets_iterator& operator=(sizes_to_offsets_iterator&&) = default;

 protected:
  template <typename S, typename R>
  friend sizes_to_offsets_iterator<S, R> make_sizes_to_offsets_iterator(S, S, R*);

  /**
   * @brief Iterator constructor
   *
   * Use the make_sizes_to_offsets_iterator() to create an instance of this class
   */
  sizes_to_offsets_iterator(ScanIterator begin, ScanIterator end, LastType* last)
    : itr_{begin}, end_{thrust::prev(end)}, last_{last}
  {
  }

  ScanIterator itr_{};
  ScanIterator end_{};
  LastType* last_{};
};

/**
 * @brief Create an instance of a sizes_to_offsets_iterator
 *
 * @code{.pseudo}
 *  auto begin = // begin input iterator
 *  auto end = // end input iterator
 *  auto result = rmm::device_uvector(std::distance(begin,end), stream);
 *  auto last = rmm::device_scalar<int64_t>(0, stream);
 *  auto itr = make_sizes_to_offsets_iterator(result.begin(),
 *                                            result.end(),
 *                                            last.data());
 *  thrust::exclusive_scan(rmm::exec_policy(stream), begin, end, itr, int64_t{0});
 *  // last contains the value of the final element in the scan result
 * @endcode
 *
 * @tparam ScanIterator Output iterator type for use in a scan operation
 * @tparam LastType Type used for holding the final element value
 *
 * @param begin Output iterator for scan
 * @param end End of the output iterator for scan
 * @param last Last element in the scan is stored here
 * @return Instance of iterator
 */
template <typename ScanIterator, typename LastType>
static sizes_to_offsets_iterator<ScanIterator, LastType> make_sizes_to_offsets_iterator(
  ScanIterator begin, ScanIterator end, LastType* last)
{
  return sizes_to_offsets_iterator<ScanIterator, LastType>(begin, end, last);
}

/**
 * @brief Perform an exclusive-scan and capture the final element value
 *
 * This performs an exclusive-scan (addition only) on the given input `[begin, end)`.
 * The output of the scan is placed in `result` and the value of the last element is returned.
 *
 * This implementation will return the last element in `int64_t` or `uint64_t` precision
 * as appropriate regardless of the input or result types.
 * This can be used to check if the scan operation overflowed when the input and result are
 * declared as smaller types.
 *
 * Only integral types for input and result types are supported.
 *
 * Note that `begin == result` is allowed but `result` may not overlap `[begin,end)` otherwise the
 * behavior is undefined.
 *
 * @code{.pseudo}
 *   auto const bytes = cudf::detail::sizes_to_offsets(
 *     d_offsets, d_offsets + strings_count + 1, d_offsets, stream);
 *   CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
 *               "Size of output exceeds column size limit");
 * @endcode
 *
 * @tparam SizesIterator Iterator type for input and output of the scan using addition operation
 *
 * @param begin Input iterator for scan
 * @param end End of the input iterator
 * @param result Output iterator for scan result
 * @return The last element of the scan
 */
template <typename SizesIterator>
auto sizes_to_offsets(SizesIterator begin,
                      SizesIterator end,
                      SizesIterator result,
                      rmm::cuda_stream_view stream)
{
  using SizeType = typename thrust::iterator_traits<SizesIterator>::value_type;
  static_assert(std::is_integral_v<SizeType>,
                "Only numeric types are supported by sizes_to_offsets");

  using LastType    = std::conditional_t<std::is_signed_v<SizeType>, int64_t, uint64_t>;
  auto last_element = rmm::device_scalar<LastType>(0, stream);
  auto output_itr =
    make_sizes_to_offsets_iterator(result, result + std::distance(begin, end), last_element.data());
  // This function uses the type of the initialization parameter as the accumulator type
  // when computing the individual scan output elements.
  thrust::exclusive_scan(rmm::exec_policy(stream), begin, end, output_itr, LastType{0});
  return last_element.value(stream);
}

}  // namespace detail
}  // namespace cudf
