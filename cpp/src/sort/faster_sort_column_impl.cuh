/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "common_sort_impl.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

namespace cudf {
namespace detail {

template <typename F>
struct float_pair {
  size_type bnan;
  F f;
};

template <typename F>
struct float_decomposer {
  __device__ cuda::std::tuple<size_type&, F&> operator()(float_pair<F>& key) const
  {
    return {key.bnan, key.f};
  }
};

template <typename F>
struct float_to_pair_and_seq {
  F* fs;
  __device__ cuda::std::pair<float_pair<F>, size_type> operator()(cudf::size_type idx)
  {
    auto f = fs[idx];
    auto s = (isnan(f) * idx);  // makes stable
    return {float_pair<F>{s, f}, idx};
  }
};

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for most primitive types.
 *
 * @tparam method Whether to use stable sort
 * @param input Column to sort. The column data is not modified.
 * @param indices The result of the sort
 * @param ascending Sort order
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <sort_method method>
void faster_sorted_order(column_view const& input,
                         mutable_column_view& indices,
                         bool ascending,
                         rmm::cuda_stream_view stream);

template <sort_method method>
struct faster_sorted_order_fn {
  /**
   * @brief Sorts fixed-width columns using faster thrust sort
   *
   * Should not be called if `input.has_nulls()==true`
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void faster_sort(mutable_column_view& input,
                   mutable_column_view& indices,
                   bool ascending,
                   rmm::cuda_stream_view stream)
  {
    // A thrust sort on a column of most primitive types will use a radix sort.
    // For other fixed-width types, thrust may use merge-sort.
    // The API sorts inplace so it requires making a copy of the input data
    // and creating the input indices sequence.

    auto const do_sort = [&](auto const comp) {
      if constexpr (method == sort_method::STABLE) {
        thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                                   input.begin<T>(),
                                   input.end<T>(),
                                   indices.begin<size_type>(),
                                   comp);
      } else {
        thrust::sort_by_key(rmm::exec_policy_nosync(stream),
                            input.begin<T>(),
                            input.end<T>(),
                            indices.begin<size_type>(),
                            comp);
      }
    };

    if (ascending) {
      do_sort(cuda::std::less<T>{});
    } else {
      do_sort(cuda::std::greater<T>{});
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_floating_point<T>())>
  void operator()(mutable_column_view& input,
                  mutable_column_view& indices,  // no longer preload this
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    auto pair_in  = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_in     = pair_in.begin();
    auto pair_out = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_out    = pair_out.begin();
    auto vals     = rmm::device_uvector<size_type>(indices.size(), stream);
    auto dv_in    = vals.begin();
    auto dv_out   = indices.begin<cudf::size_type>();

    auto zip_out = thrust::make_zip_iterator(d_in, dv_in);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::counting_iterator<size_type>(0),
                      thrust::counting_iterator<size_type>(input.size()),
                      zip_out,
                      float_to_pair_and_seq<T>{input.begin<T>()});

    auto const decomposer = float_decomposer<T>{};
    auto const end_bit    = sizeof(float_pair<T>) * 8;
    auto const sv         = stream.value();
    auto const n          = input.size();
    // cub radix sort implementation is always stable
    std::size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
      nullptr, tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    if (ascending) {
      cub::DeviceRadixSort::SortPairs(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>() && !cudf::is_floating_point<T>())>
  void operator()(mutable_column_view& input,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    faster_sort<T>(input, indices, ascending, stream);
  }

  template <typename T, CUDF_ENABLE_IF(not cudf::is_fixed_width<T>())>
  void operator()(mutable_column_view&, mutable_column_view&, bool, rmm::cuda_stream_view)
  {
    CUDF_UNREACHABLE("invalid type for faster sort");
  }
};

}  // namespace detail
}  // namespace cudf
