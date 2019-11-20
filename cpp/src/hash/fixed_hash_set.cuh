/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <utilities/device_atomics.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <hash/helper_functions.cuh>

namespace cudf {

/*
 *  fixed set on a device
 */
template <typename Element, typename Hasher = default_hash<Element>,
          typename Equality = equal_to<Element>>
class device_fixed_hash_set {
public:
  device_fixed_hash_set(size_type hash_size, const size_type *hash_begin, const Element *hash_data)
    : hash_size{hash_size}, hash_begin{hash_begin}, hash_data{hash_data}, hasher(), equals() {
  }

  bool __device__ contains(Element e) const {
    size_type loc = hasher(e) % (2 * hash_size);

    for (size_type i = hash_begin[loc] ; i < hash_begin[loc+1] ; ++i) {
      if (equals(hash_data[i],e))
        return true;
    }

    return false;
  }

private:
  Hasher           hasher;
  Equality         equals;
  size_type        hash_size;
  const size_type *hash_begin;
  const Element   *hash_data;
};

/*
 * Fixed size set on a device.  
 *
 */
template <typename Element, typename Hasher = default_hash<Element>,
          typename Equality = equal_to<Element>>
class fixed_hash_set {
public:
  /**---------------------------------------------------------------------------*
   * @brief Factory to construct a new fixed_device_set
   *---------------------------------------------------------------------------**/
  static fixed_hash_set<Element> create(column_view const& col,
                                        cudaStream_t stream) {

    auto d_column = column_device_view::create(col);
    auto d_col = *d_column;

    rmm::device_vector<size_type> hash_bins_start(2 * d_col.size() + 1, size_type{0});
    rmm::device_vector<size_type> hash_bins_end(2 * d_col.size() + 1, size_type{0});
    rmm::device_vector<Element>   hash_data(d_col.size());

    Hasher hasher;
    size_type *d_hash_bins_start = thrust::raw_pointer_cast(hash_bins_start.data());
    size_type *d_hash_bins_end = thrust::raw_pointer_cast(hash_bins_end.data());
    Element *d_hash_data = thrust::raw_pointer_cast(hash_data.data());

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(col.size()),
                     [d_hash_bins_start, d_col, hasher] __device__ (size_t idx) {
                       if (!d_col.is_null(idx)) {
                         Element e = d_col.element<Element>(idx);
                         size_type tmp = hasher(e) % (2 * d_col.size());
                         atomicAdd(d_hash_bins_start + tmp, size_type{1});
                       }
                     });

    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           hash_bins_start.begin(),
                           hash_bins_start.end(),
                           hash_bins_end.begin());

    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 hash_bins_end.begin(),
                 hash_bins_end.end(),
                 hash_bins_start.begin());

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(col.size()),
                     [d_hash_bins_end, d_hash_data, d_col, hasher] __device__ (size_t idx) {
                       if (!d_col.is_null(idx)) {
                         Element e = d_col.element<Element>(idx);
                         size_type tmp = hasher(e) % (2 * d_col.size());
                         size_type offset = atomicAdd(d_hash_bins_end + tmp, size_type{1});
                         d_hash_data[offset] = e;
                       }
                     });

    return fixed_hash_set(d_col.size(), std::move(hash_bins_start), std::move(hash_data));
  }

  device_fixed_hash_set<Element, Hasher, Equality> to_device() {
    return device_fixed_hash_set<Element, Hasher, Equality>(size,
                                                       thrust::raw_pointer_cast(hash_bins.data()),
                                                       thrust::raw_pointer_cast(hash_data.data()));
  }

private:
  fixed_hash_set(size_type size, rmm::device_vector<size_type> &&hash_bins, rmm::device_vector<Element> &&hash_data)
    : size{size}, hash_bins{std::move(hash_bins)}, hash_data{std::move(hash_data)} {}
  
  size_type size;
  rmm::device_vector<size_type> hash_bins;
  rmm::device_vector<Element> hash_data;
};

}
