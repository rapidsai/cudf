/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/aggregation.hpp>
#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <rmm/device_scalar.hpp>

namespace cudf {

namespace experimental {

namespace { // anonymous

  /**
   * @brief Computes the rolling window function
   *
   * @tparam ColumnType  Datatype of values pointed to by the pointers
   * @tparam agg_op  A functor that defines the aggregation operation
   * @tparam is_mean Compute mean=sum/count across all valid elements in the window
   * @tparam block_size CUDA block size for the kernel
   * @tparam has_nulls true if the input column has nulls
   * @tparam PrecedingWindowIterator iterator type, for `preceding` offsets (inferred)
   * @tparam FollowingWindowIterator iterator type, for `following` offsets (inferred)
   * @param input Input column device view
   * @param output Output column device view
   * @param preceding_window_begin[in] Rolling window size iterator, accumulates from
   *                in_col[i-preceding_window] to in_col[i] inclusive
   * @param following_window_begin[in] Rolling window size iterator in the forward
   *                direction, accumulates from in_col[i] to
   *                in_col[i+following_window] inclusive
   * @param min_periods[in]  Minimum number of observations in window required to
   *                have a value, otherwise 0 is stored in the valid bit mask
   */
  template <typename T, typename agg_op, aggregation::Kind op, int block_size, bool has_nulls,
            typename PrecedingWindowIterator, typename FollowingWindowIterator>
  __launch_bounds__(block_size)
  __global__
  void gpu_rolling(column_device_view input,
                  mutable_column_device_view output,
                  size_type * __restrict__ output_valid_count,
                  PrecedingWindowIterator preceding_window_begin,
                  FollowingWindowIterator following_window_begin,
                  size_type min_periods)
  {
    size_type i = blockIdx.x * block_size + threadIdx.x;
    size_type stride = block_size * gridDim.x;

    size_type warp_valid_count{0};

    auto active_threads = __ballot_sync(0xffffffff, i < input.size());
    while(i < input.size())
    {
      T val = agg_op::template identity<T>();
      // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
      // for CUDA 10.0 and below (fixed in CUDA 10.1)
      volatile cudf::size_type count = 0;

      size_type preceding_window = preceding_window_begin[i];
      size_type following_window = following_window_begin[i];

      // compute bounds
      size_type start_index = max(0, i - preceding_window);
      size_type end_index = min(input.size(), i + following_window + 1); // exclusive

      // aggregate
      // TODO: We should explore using shared memory to avoid redundant loads.
      //       This might require separating the kernel into a special version
      //       for dynamic and static sizes.
      for (size_type j = start_index; j < end_index; j++) {
        if (!has_nulls || input.is_valid(j)) {
          // Element type and output type are different for COUNT
          T element = (op == aggregation::COUNT) ? T{0} : input.element<T>(j);
          val = agg_op{}(element, val);
          count++;
        }
      }

      // check if we have enough input samples
      bool output_is_valid = (count >= min_periods);

      // set the mask
      cudf::bitmask_type result_mask{__ballot_sync(active_threads, output_is_valid)};

      // only one thread writes the mask
      if (0 == threadIdx.x % cudf::experimental::detail::warp_size) {
        output.set_mask_word(cudf::word_index(i), result_mask);
        warp_valid_count += __popc(result_mask);
      }

      // store the output value, one per thread
      if (output_is_valid)
        cudf::detail::store_output_functor<T, op == aggregation::MEAN>{}(output.element<T>(i),
                                                                              val, count);
      // process next element 
      i += stride;
      active_threads = __ballot_sync(active_threads, i < input.size());
    }

    // sum the valid counts across the whole block  
    size_type block_valid_count = 
      cudf::experimental::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    
    if(threadIdx.x == 0) {
      atomicAdd(output_valid_count, block_valid_count);
    }
  }

  template <typename InputType>
  struct rolling_window_launcher
  {
    template <typename T, typename agg_op, aggregation::Kind op, typename PrecedingWindowIterator, typename FollowingWindowIterator>
    std::enable_if_t<cudf::detail::is_supported<T, agg_op,
                                    op, op == aggregation::MEAN>(), std::unique_ptr<column>>
    launch(column_view const& input,
          PrecedingWindowIterator preceding_window_begin,
          FollowingWindowIterator following_window_begin,
          size_type min_periods,
          std::unique_ptr<aggregation> const& aggr,
          rmm::mr::device_memory_resource *mr,
          cudaStream_t stream) {

      if (input.is_empty()) return empty_like(input);

      cudf::nvtx::range_push("CUDF_ROLLING_WINDOW", cudf::nvtx::color::ORANGE);

      // output is always nullable, COUNT always INT32 output
      std::unique_ptr<column> output = (op == aggregation::COUNT) ?
          make_numeric_column(cudf::data_type{cudf::INT32}, input.size(),
                              cudf::UNINITIALIZED, stream, mr) :
          cudf::experimental::detail::allocate_like(input, input.size(),
            cudf::experimental::mask_allocation_policy::ALWAYS, mr, stream);

      constexpr cudf::size_type block_size = 256;
      cudf::experimental::detail::grid_1d grid(input.size(), block_size);

      auto input_device_view = column_device_view::create(input, stream);
      auto output_device_view = mutable_column_device_view::create(*output, stream);

      rmm::device_scalar<size_type> device_valid_count{0, stream};

      if (input.has_nulls()) {
          if (op == aggregation::COUNT) {
              gpu_rolling<size_type, agg_op, op, block_size, true><<<grid.num_blocks, block_size, 0, stream>>>
                  (*input_device_view, *output_device_view, device_valid_count.data(),
                  preceding_window_begin, following_window_begin, min_periods);
          }
          else {
              gpu_rolling<InputType, agg_op, op, block_size, true><<<grid.num_blocks, block_size, 0, stream>>>
                  (*input_device_view, *output_device_view, device_valid_count.data(),
                  preceding_window_begin, following_window_begin, min_periods);
          }
      } else {
          if (op == aggregation::COUNT) {
              gpu_rolling<size_type, agg_op, op, block_size, false><<<grid.num_blocks, block_size, 0, stream>>>
                  (*input_device_view, *output_device_view, device_valid_count.data(),
                  preceding_window_begin, following_window_begin, min_periods);
          }
          else {
              gpu_rolling<InputType, agg_op, op, block_size, false><<<grid.num_blocks, block_size, 0, stream>>>
                  (*input_device_view, *output_device_view, device_valid_count.data(),
                  preceding_window_begin, following_window_begin, min_periods);
          }
      }

      output->set_null_count(output->size() - device_valid_count.value(stream));

      // check the stream for debugging
      CHECK_CUDA(stream);

      cudf::nvtx::range_pop();

      return std::move(output);
    }

    template <typename T, typename agg_op, aggregation::Kind op, typename PrecedingWindowIterator, typename FollowingWindowIterator>
    std::enable_if_t<!cudf::detail::is_supported<T, agg_op,
                                    op, op == aggregation::MEAN>(), std::unique_ptr<column>>
    launch (column_view const& input,
            PrecedingWindowIterator preceding_window_begin,
            FollowingWindowIterator following_window_begin,
            size_type min_periods,
            std::unique_ptr<aggregation> const& aggr,
            rmm::mr::device_memory_resource *mr,
            cudaStream_t stream) {
        CUDF_FAIL("Aggregation operator and/or input type combination is invalid");
    }

    template<aggregation::Kind op, typename PrecedingWindowIterator, typename FollowingWindowIterator>
    std::enable_if_t<!(op == aggregation::MEAN), std::unique_ptr<column>>
    operator()(column_view const& input,
                                      PrecedingWindowIterator preceding_window_begin,
                                      FollowingWindowIterator following_window_begin,
                                      size_type min_periods,
                                      std::unique_ptr<aggregation> const& aggr,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream)
    {
        return launch <InputType, typename cudf::experimental::detail::corresponding_operator<op>::type, op, PrecedingWindowIterator, FollowingWindowIterator> (
                input,
                preceding_window_begin,
                following_window_begin,
                min_periods,
                aggr,
                mr,
                stream);
    }

    template<aggregation::Kind op, typename PrecedingWindowIterator, typename FollowingWindowIterator>
    std::enable_if_t<(op == aggregation::MEAN), std::unique_ptr<column>>
    operator()(column_view const& input,
                                      PrecedingWindowIterator preceding_window_begin,
                                      FollowingWindowIterator following_window_begin,
                                      size_type min_periods,
                                      std::unique_ptr<aggregation> const& aggr,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream) {

        return launch <InputType, cudf::DeviceSum, op, PrecedingWindowIterator, FollowingWindowIterator> (
                input,
                preceding_window_begin,
                following_window_begin,
                min_periods,
                aggr,
                mr,
                stream);
    }
  };

  struct dispatch_rolling {
      template <typename T, typename PrecedingWindowIterator, typename FollowingWindowIterator>
      std::unique_ptr<column> operator()(column_view const& input,
                                      PrecedingWindowIterator preceding_window_begin,
                                      FollowingWindowIterator following_window_begin,
                                      size_type min_periods,
                                      std::unique_ptr<aggregation> const& aggr,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream) {

          return cudf::experimental::detail::aggregation_dispatcher(aggr->kind, rolling_window_launcher<T>{},
                                        input,
                                        preceding_window_begin, following_window_begin,
                                        min_periods, aggr, mr, stream);
      }
  };
} // namespace anonymous;

namespace detail {

  // Applies a rolling window function to the values in a column.
  template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
  std::unique_ptr<column> rolling_window(column_view const& input,
                                        PrecedingWindowIterator preceding_window_begin,
                                        FollowingWindowIterator following_window_begin,
                                        size_type min_periods,
                                        std::unique_ptr<aggregation> const& aggr,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream = 0)
  {
    return cudf::experimental::type_dispatcher(input.type(),
                                              dispatch_rolling{},
                                              input, preceding_window_begin, following_window_begin,
                                              min_periods, aggr, mr, stream);
  }
} // namespace detail;
} // namespace experimental;
} // namespace cudf;
