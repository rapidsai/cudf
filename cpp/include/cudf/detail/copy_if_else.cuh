/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <cuda/std/optional>
#include <thrust/iterator/iterator_traits.h>

namespace cudf {
namespace detail {
namespace {  // anonymous

template <size_type block_size,
          typename T,
          typename LeftIter,
          typename RightIter,
          typename Filter,
          bool has_nulls>
__launch_bounds__(block_size) CUDF_KERNEL
  void copy_if_else_kernel(LeftIter lhs,
                           RightIter rhs,
                           Filter filter,
                           mutable_column_device_view out,
                           size_type* __restrict__ const valid_count)
{
  size_type const tid            = threadIdx.x + blockIdx.x * block_size;
  int const warp_id              = tid / warp_size;
  size_type const warps_per_grid = gridDim.x * block_size / warp_size;

  // begin/end indices for the column data
  size_type begin = 0;
  size_type end   = out.size();
  // warp indices.  since 1 warp == 32 threads == sizeof(bitmask_type) * 8,
  // each warp will process one (32 bit) of the validity mask via
  // __ballot_sync()
  size_type warp_begin = cudf::word_index(begin);
  size_type warp_end   = cudf::word_index(end - 1);

  // lane id within the current warp
  constexpr size_type leader_lane{0};
  int const lane_id = threadIdx.x % warp_size;

  size_type warp_valid_count{0};

  // current warp.
  size_type warp_cur = warp_begin + warp_id;
  size_type index    = tid;
  while (warp_cur <= warp_end) {
    auto const opt_value =
      (index < end) ? (filter(index) ? lhs[index] : rhs[index]) : cuda::std::nullopt;
    if (opt_value) { out.element<T>(index) = static_cast<T>(*opt_value); }

    // update validity
    if (has_nulls) {
      // the final validity mask for this warp
      int warp_mask = __ballot_sync(0xFFFF'FFFFu, opt_value.has_value());
      // only one guy in the warp needs to update the mask and count
      if (lane_id == 0) {
        out.set_mask_word(warp_cur, warp_mask);
        warp_valid_count += __popc(warp_mask);
      }
    }

    // next grid
    warp_cur += warps_per_grid;
    index += block_size * gridDim.x;
  }

  if (has_nulls) {
    // sum all null counts across all warps
    size_type block_valid_count =
      single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
    // block_valid_count will only be valid on thread 0
    if (threadIdx.x == 0) {
      // using an atomic here because there are multiple blocks doing this work
      atomicAdd(valid_count, block_valid_count);
    }
  }
}

}  // anonymous namespace

/**
 * @brief Returns a new column, where each element is selected from either of two input ranges based
 * on a filter
 *
 * Given two ranges lhs and rhs, and a unary filter function, this function will allocate and return
 * an output column that contains `lhs[i]` if `function(i) == true` or `rhs[i]` if `function(i) ==
 * false`. The validity of the elements is propagated to the output.
 *
 * The range lhs is defined by iterators `[lhs_begin, lhs_end)`. The `size` of output is
 * determined by the distance between `lhs_begin` and `lhs_end`.
 *
 * The range rhs is defined by `[rhs, rhs + size)`
 *
 * Example:
 * @code{.pseudo}
 * lhs = {1, 2, 3, -, 5}
 * rhs = {-, 6, 7, 8, 9}
 *
 * filter = [](i) {
 *   bool arr[5] = {1, 1, 0, 1, 0}
 *   return arr[i];
 * }
 *
 * output = {1, 2, 7, -, 9}
 * @endcode
 *
 * @tparam FilterFn   A function of type `bool(size_type)`
 * @tparam LeftIter   An iterator of pair type where `first` is the value and `second` is the
 *                    validity
 * @tparam RightIter  An iterator of pair type where `first` is the value and `second` is the
 *                    validity
 * @param nullable    Indicate whether either input range can contain nulls
 * @param lhs_begin   Begin iterator of lhs range
 * @param lhs_end     End iterator of lhs range
 * @param rhs         Begin iterator of rhs range
 * @param filter      Function of type `FilterFn` which determines for index `i` where to get the
 *                    corresponding output value from
 * @param out_type    `cudf::data_type` of the returned column
 * @param stream      CUDA stream used for device memory operations and kernel launches.
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            A new column that contains the values from either `lhs` or `rhs` as determined
 *                    by `filter[i]`
 */
template <typename FilterFn, typename LeftIter, typename RightIter>
std::unique_ptr<column> copy_if_else(bool nullable,
                                     LeftIter lhs_begin,
                                     LeftIter lhs_end,
                                     RightIter rhs,
                                     FilterFn filter,
                                     cudf::data_type output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  // This is the type of the thrust::optional element in the passed iterators
  using Element = typename thrust::iterator_traits<LeftIter>::value_type::value_type;

  size_type size           = std::distance(lhs_begin, lhs_end);
  size_type num_els        = cudf::util::round_up_safe(size, warp_size);
  constexpr int block_size = 256;
  cudf::detail::grid_1d grid{num_els, block_size, 1};

  std::unique_ptr<column> out = make_fixed_width_column(
    output_type, size, nullable ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED, stream, mr);

  auto out_v = mutable_column_device_view::create(*out, stream);

  // if we have validity in the output
  if (nullable) {
    rmm::device_scalar<size_type> valid_count{0, stream};

    // call the kernel
    copy_if_else_kernel<block_size, Element, LeftIter, RightIter, FilterFn, true>
      <<<grid.num_blocks, block_size, 0, stream.value()>>>(
        lhs_begin, rhs, filter, *out_v, valid_count.data());

    out->set_null_count(size - valid_count.value(stream));
  } else {
    // call the kernel
    copy_if_else_kernel<block_size, Element, LeftIter, RightIter, FilterFn, false>
      <<<grid.num_blocks, block_size, 0, stream.value()>>>(lhs_begin, rhs, filter, *out_v, nullptr);
  }

  return out;
}

}  // namespace detail

}  // namespace cudf
