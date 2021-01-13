/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rolling/jit/code/code.h>
#include <rolling/rolling_detail.hpp>
#include <rolling/rolling_jit_detail.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <jit/launcher.h>
#include <jit/parser.h>
#include <jit/type.h>
#include <jit/bit.hpp.jit>
#include <jit/rolling_jit_detail.hpp.jit>
#include <jit/types.hpp.jit>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/binary_search.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <memory>

namespace cudf {

namespace detail {
namespace {  // anonymous
/**
 * @brief Only COUNT_VALID operation is executed and count is updated
 *        depending on `min_periods` and returns true if it was
 *        valid, else false.
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls,
          std::enable_if_t<op == aggregation::COUNT_VALID>* = nullptr>
bool __device__ process_rolling_window(column_device_view input,
                                       column_device_view ignored_default_outputs,
                                       mutable_column_device_view output,
                                       size_type start_index,
                                       size_type end_index,
                                       size_type current_index,
                                       size_type min_periods)
{
  // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
  // for CUDA 10.0 and below (fixed in CUDA 10.1)
  volatile cudf::size_type count = 0;

  bool output_is_valid = ((end_index - start_index) >= min_periods);

  if (output_is_valid) {
    if (!has_nulls) {
      count = end_index - start_index;
    } else {
      count = thrust::count_if(thrust::seq,
                               thrust::make_counting_iterator(start_index),
                               thrust::make_counting_iterator(end_index),
                               [&input](auto i) { return input.is_valid_nocheck(i); });
    }
    output.element<OutputType>(current_index) = count;
  }

  return output_is_valid;
}

/**
 * @brief Only COUNT_ALL operation is executed and count is updated
 *        depending on `min_periods` and returns true if it was
 *        valid, else false.
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls,
          std::enable_if_t<op == aggregation::COUNT_ALL>* = nullptr>
bool __device__ process_rolling_window(column_device_view input,
                                       column_device_view ignored_default_outputs,
                                       mutable_column_device_view output,
                                       size_type start_index,
                                       size_type end_index,
                                       size_type current_index,
                                       size_type min_periods)
{
  cudf::size_type count = end_index - start_index;

  bool output_is_valid                      = count >= min_periods;
  output.element<OutputType>(current_index) = count;

  return output_is_valid;
}

/**
 * @brief Calculates row-number of current index within [start_index, end_index). Count is updated
 *        depending on `min_periods`. Returns `true` if it was valid, else `false`.
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls,
          std::enable_if_t<op == aggregation::ROW_NUMBER>* = nullptr>
bool __device__ process_rolling_window(column_device_view input,
                                       column_device_view ignored_default_outputs,
                                       mutable_column_device_view output,
                                       size_type start_index,
                                       size_type end_index,
                                       size_type current_index,
                                       size_type min_periods)
{
  bool output_is_valid                      = end_index - start_index >= min_periods;
  output.element<OutputType>(current_index) = current_index - start_index + 1;

  return output_is_valid;
}

/**
 * @brief LEAD(N): Returns the row from the input column, at the specified offset past the
 *        current row.
 * If the offset crosses the grouping boundary or column boundary for
 * a given row, a "default" value is returned. The "default" value is null, by default.
 *
 * E.g. Consider an input column with the following values and grouping:
 *      [10, 11, 12, 13,   20, 21, 22, 23]
 *      <------G1----->   <------G2------>
 *
 * LEAD(input_col, 1) yields:
 *      [11, 12, 13, null,  21, 22, 23, null]
 *
 * LEAD(input_col, 1, 99) (where 99 indicates the default) yields:
 *      [11, 12, 13, 99,  21, 22, 23, 99]
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls>
std::enable_if_t<(op == aggregation::LEAD) && (cudf::is_fixed_width<InputType>()), bool> __device__
process_rolling_window(column_device_view input,
                       column_device_view default_outputs,
                       mutable_column_device_view output,
                       size_type start_index,
                       size_type end_index,
                       size_type current_index,
                       size_type min_periods,
                       agg_op device_agg_op)
{
  // Offsets have already been normalized.
  auto row_offset = device_agg_op.row_offset;

  // Check if row is invalid.
  if (row_offset > (end_index - current_index - 1)) {
    // Invalid row marked. Use default value, if available.
    if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

    output.element<OutputType>(current_index) = default_outputs.element<OutputType>(current_index);
    return true;
  }

  // Not an invalid row.
  auto index   = current_index + row_offset;
  auto is_null = input.is_null(index);
  if (!is_null) { output.element<OutputType>(current_index) = input.element<InputType>(index); }
  return !is_null;
}

/**
 * @brief LAG(N): returns the row from the input column at the specified offset preceding
 *        the current row.
 * If the offset crosses the grouping boundary or column boundary for
 * a given row, a "default" value is returned. The "default" value is null, by default.
 *
 * E.g. Consider an input column with the following values and grouping:
 *      [10, 11, 12, 13,   20, 21, 22, 23]
 *      <------G1----->   <------G2------>
 *
 * LAG(input_col, 2) yields:
 *      [null, null, 10, 11, null, null, 20, 21]
 * LAG(input_col, 2, 99) yields:
 *      [99, 99, 10, 11, 99, 99, 20, 21]
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls>
std::enable_if_t<(op == aggregation::LAG) && (cudf::is_fixed_width<InputType>()), bool> __device__
process_rolling_window(column_device_view input,
                       column_device_view default_outputs,
                       mutable_column_device_view output,
                       size_type start_index,
                       size_type end_index,
                       size_type current_index,
                       size_type min_periods,
                       agg_op device_agg_op)
{
  // Offsets have already been normalized.
  auto row_offset = device_agg_op.row_offset;

  // Check if row is invalid.
  if (row_offset > (current_index - start_index)) {
    // Invalid row marked. Use default value, if available.
    if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

    output.element<OutputType>(current_index) = default_outputs.element<OutputType>(current_index);
    return true;
  }

  // Not an invalid row.
  auto index   = current_index - row_offset;
  auto is_null = input.is_null(index);
  if (!is_null) { output.element<OutputType>(current_index) = input.element<InputType>(index); }
  return !is_null;
}

/**
 * @brief Only used for `string_view` type to get ARGMIN and ARGMAX, which
 *        will be used to gather MIN and MAX. And returns true if the
 *        operation was valid, else false.
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls,
          std::enable_if_t<(op == aggregation::ARGMIN or op == aggregation::ARGMAX) and
                           std::is_same<InputType, cudf::string_view>::value>* = nullptr>
bool __device__ process_rolling_window(column_device_view input,
                                       column_device_view ignored_default_outputs,
                                       mutable_column_device_view output,
                                       size_type start_index,
                                       size_type end_index,
                                       size_type current_index,
                                       size_type min_periods)
{
  // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
  // for CUDA 10.0 and below (fixed in CUDA 10.1)
  volatile cudf::size_type count = 0;
  InputType val                  = agg_op::template identity<InputType>();
  OutputType val_index           = (op == aggregation::ARGMIN) ? ARGMIN_SENTINEL : ARGMAX_SENTINEL;

  for (size_type j = start_index; j < end_index; j++) {
    if (!has_nulls || input.is_valid(j)) {
      InputType element = input.element<InputType>(j);
      val               = agg_op{}(element, val);
      if (val == element) { val_index = j; }
      count++;
    }
  }

  bool output_is_valid = (count >= min_periods);
  // -1 will help identify null elements while gathering for Min and Max
  // In case of count, this would be null, so doesn't matter.
  output.element<OutputType>(current_index) = (output_is_valid) ? val_index : -1;

  // The gather mask shouldn't contain null values, so
  // always return zero
  return true;
}

/**
 * @brief Operates on only fixed-width types and returns true if the
 *        operation was valid, else false.
 */
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          bool has_nulls,
          std::enable_if_t<!std::is_same<InputType, cudf::string_view>::value and
                           !(op == aggregation::COUNT_VALID || op == aggregation::COUNT_ALL ||
                             op == aggregation::ROW_NUMBER || op == aggregation::LEAD ||
                             op == aggregation::LAG)>* = nullptr>
bool __device__ process_rolling_window(column_device_view input,
                                       column_device_view ignored_default_outputs,
                                       mutable_column_device_view output,
                                       size_type start_index,
                                       size_type end_index,
                                       size_type current_index,
                                       size_type min_periods)
{
  // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
  // for CUDA 10.0 and below (fixed in CUDA 10.1)
  volatile cudf::size_type count = 0;
  OutputType val                 = agg_op::template identity<OutputType>();

  for (size_type j = start_index; j < end_index; j++) {
    if (!has_nulls || input.is_valid(j)) {
      OutputType element = input.element<InputType>(j);
      val                = agg_op{}(element, val);
      count++;
    }
  }

  bool output_is_valid = (count >= min_periods);

  // store the output value, one per thread
  cudf::detail::rolling_store_output_functor<OutputType, op == aggregation::MEAN>{}(
    output.element<OutputType>(current_index), val, count);

  return output_is_valid;
}

/**
 * @brief Computes the rolling window function
 *
 * @tparam InputType  Datatype of `input`
 * @tparam OutputType  Datatype of `output`
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam op The aggregation operator (enum value)
 * @tparam block_size CUDA block size for the kernel
 * @tparam has_nulls true if the input column has nulls
 * @tparam PrecedingWindowIterator iterator type (inferred)
 * @tparam FollowingWindowIterator iterator type (inferred)
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
template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          int block_size,
          bool has_nulls,
          typename PrecedingWindowIterator,
          typename FollowingWindowIterator>
__launch_bounds__(block_size) __global__
  void gpu_rolling(column_device_view input,
                   column_device_view default_outputs,
                   mutable_column_device_view output,
                   size_type* __restrict__ output_valid_count,
                   PrecedingWindowIterator preceding_window_begin,
                   FollowingWindowIterator following_window_begin,
                   size_type min_periods)
{
  size_type i      = blockIdx.x * block_size + threadIdx.x;
  size_type stride = block_size * gridDim.x;

  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < input.size());
  while (i < input.size()) {
    size_type preceding_window = preceding_window_begin[i];
    size_type following_window = following_window_begin[i];

    // compute bounds
    size_type start       = min(input.size(), max(0, i - preceding_window + 1));
    size_type end         = min(input.size(), max(0, i + following_window + 1));
    size_type start_index = min(start, end);
    size_type end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.

    volatile bool output_is_valid = false;
    output_is_valid = process_rolling_window<InputType, OutputType, agg_op, op, has_nulls>(
      input, default_outputs, output, start_index, end_index, i, min_periods);

    // set the mask
    cudf::bitmask_type result_mask{__ballot_sync(active_threads, output_is_valid)};

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.size());
  }

  // sum the valid counts across the whole block
  size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

template <typename InputType,
          typename OutputType,
          typename agg_op,
          aggregation::Kind op,
          int block_size,
          bool has_nulls,
          typename PrecedingWindowIterator,
          typename FollowingWindowIterator>
__launch_bounds__(block_size) __global__
  void gpu_rolling(column_device_view input,
                   column_device_view default_outputs,
                   mutable_column_device_view output,
                   size_type* __restrict__ output_valid_count,
                   PrecedingWindowIterator preceding_window_begin,
                   FollowingWindowIterator following_window_begin,
                   size_type min_periods,
                   agg_op device_agg_op)
{
  size_type i      = blockIdx.x * block_size + threadIdx.x;
  size_type stride = block_size * gridDim.x;

  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < input.size());
  while (i < input.size()) {
    size_type preceding_window = preceding_window_begin[i];
    size_type following_window = following_window_begin[i];

    // compute bounds
    size_type start       = min(input.size(), max(0, i - preceding_window + 1));
    size_type end         = min(input.size(), max(0, i + following_window + 1));
    size_type start_index = min(start, end);
    size_type end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.

    volatile bool output_is_valid = false;
    output_is_valid = process_rolling_window<InputType, OutputType, agg_op, op, has_nulls>(
      input, default_outputs, output, start_index, end_index, i, min_periods, device_agg_op);

    // set the mask
    cudf::bitmask_type result_mask{__ballot_sync(active_threads, output_is_valid)};

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.size());
  }

  // sum the valid counts across the whole block
  size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

template <typename InputType>
struct rolling_window_launcher {
  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  size_type kernel_launcher(column_view const& input,
                            column_view const& default_outputs,
                            mutable_column_view& output,
                            PrecedingWindowIterator preceding_window_begin,
                            FollowingWindowIterator following_window_begin,
                            size_type min_periods,
                            std::unique_ptr<aggregation> const& agg,
                            rmm::cuda_stream_view stream)
  {
    using Type    = device_storage_type_t<T>;
    using OutType = device_storage_type_t<target_type_t<InputType, op>>;

    constexpr cudf::size_type block_size = 256;
    cudf::detail::grid_1d grid(input.size(), block_size);

    auto input_device_view           = column_device_view::create(input, stream);
    auto output_device_view          = mutable_column_device_view::create(output, stream);
    auto default_outputs_device_view = column_device_view::create(default_outputs, stream);

    rmm::device_scalar<size_type> device_valid_count{0, stream};

    if (input.has_nulls()) {
      gpu_rolling<Type, OutType, agg_op, op, block_size, true>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods);
    } else {
      gpu_rolling<Type, OutType, agg_op, op, block_size, false>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods);
    }

    size_type valid_count = device_valid_count.value(stream);

    // check the stream for debugging
    CHECK_CUDA(stream.value());

    return valid_count;
  }

  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  size_type kernel_launcher(column_view const& input,
                            column_view const& default_outputs,
                            mutable_column_view& output,
                            PrecedingWindowIterator preceding_window_begin,
                            FollowingWindowIterator following_window_begin,
                            size_type min_periods,
                            std::unique_ptr<aggregation> const& agg,
                            agg_op const& device_agg_op,
                            rmm::cuda_stream_view stream)
  {
    using Type    = device_storage_type_t<T>;
    using OutType = device_storage_type_t<target_type_t<InputType, op>>;

    constexpr cudf::size_type block_size = 256;
    cudf::detail::grid_1d grid(input.size(), block_size);

    auto input_device_view           = column_device_view::create(input, stream);
    auto output_device_view          = mutable_column_device_view::create(output, stream);
    auto default_outputs_device_view = column_device_view::create(default_outputs, stream);

    rmm::device_scalar<size_type> device_valid_count{0, stream};

    if (input.has_nulls()) {
      gpu_rolling<Type, OutType, agg_op, op, block_size, true>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods,
                                                             device_agg_op);
    } else {
      gpu_rolling<Type, OutType, agg_op, op, block_size, false>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods,
                                                             device_agg_op);
    }

    size_type valid_count = device_valid_count.value(stream);

    // check the stream for debugging
    CHECK_CUDA(stream.value());

    return valid_count;
  }

  // This launch is only for fixed width columns with valid aggregation option
  // numeric: All
  // timestamp: MIN, MAX, COUNT_VALID, COUNT_ALL, ROW_NUMBER
  // string, dictionary, list : COUNT_VALID, COUNT_ALL, ROW_NUMBER
  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<cudf::detail::is_rolling_supported<T, agg_op, op>() and
                     !cudf::detail::is_rolling_string_specialization<T, agg_op, op>(),
                   std::unique_ptr<column>>
  launch(column_view const& input,
         column_view const& default_outputs,
         PrecedingWindowIterator preceding_window_begin,
         FollowingWindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    if (input.is_empty()) return empty_like(input);

    auto output = make_fixed_width_column(
      target_type(input.type(), op), input.size(), mask_state::UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();
    auto valid_count =
      kernel_launcher<T, agg_op, op, PrecedingWindowIterator, FollowingWindowIterator>(
        input,
        default_outputs,
        output_view,
        preceding_window_begin,
        following_window_begin,
        min_periods,
        agg,
        stream);

    output->set_null_count(output->size() - valid_count);

    return output;
  }

  // This launch is only for string specializations
  // string: MIN, MAX
  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<cudf::detail::is_rolling_string_specialization<T, agg_op, op>(),
                   std::unique_ptr<column>>
  launch(column_view const& input,
         column_view const& default_outputs,
         PrecedingWindowIterator preceding_window_begin,
         FollowingWindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    if (input.is_empty()) return empty_like(input);

    auto output = make_numeric_column(cudf::data_type{cudf::type_to_id<size_type>()},
                                      input.size(),
                                      cudf::mask_state::UNINITIALIZED,
                                      stream,
                                      mr);

    cudf::mutable_column_view output_view = output->mutable_view();

    // Passing the agg_op and aggregation::Kind as constant to group them in pair, else it
    // evolves to error when try to use agg_op as compiler tries different combinations
    if (op == aggregation::MIN) {
      kernel_launcher<T,
                      DeviceMin,
                      aggregation::ARGMIN,
                      PrecedingWindowIterator,
                      FollowingWindowIterator>(input,
                                               default_outputs,
                                               output_view,
                                               preceding_window_begin,
                                               following_window_begin,
                                               min_periods,
                                               agg,
                                               stream);
    } else if (op == aggregation::MAX) {
      kernel_launcher<T,
                      DeviceMax,
                      aggregation::ARGMAX,
                      PrecedingWindowIterator,
                      FollowingWindowIterator>(input,
                                               default_outputs,
                                               output_view,
                                               preceding_window_begin,
                                               following_window_begin,
                                               min_periods,
                                               agg,
                                               stream);
    } else {
      CUDF_FAIL("MIN and MAX are the only supported aggregation types for string columns");
    }

    // The rows that represent null elements will be having negative values in gather map,
    // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
    auto output_table = detail::gather(table_view{{input}},
                                       output->view(),
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       detail::negative_index_policy::NOT_ALLOWED,
                                       stream,
                                       mr);
    return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
  }

  // Deals with invalid column and/or aggregation options
  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!cudf::detail::is_rolling_supported<T, agg_op, op>() and
                     !cudf::detail::is_rolling_string_specialization<T, agg_op, op>(),
                   std::unique_ptr<column>>
  launch(column_view const& input,
         column_view const& default_outputs,
         PrecedingWindowIterator preceding_window_begin,
         FollowingWindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Aggregation operator and/or input type combination is invalid");
  }

  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<cudf::is_fixed_width<T>() and
                     (op == aggregation::LEAD || op == aggregation::LAG),
                   std::unique_ptr<column>>
  launch(column_view const& input,
         column_view const& default_outputs,
         PrecedingWindowIterator preceding_window_begin,
         FollowingWindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         agg_op const& device_agg_op,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    if (input.is_empty()) return empty_like(input);

    CUDF_EXPECTS(default_outputs.type().id() == input.type().id(),
                 "Defaults column type must match input column.");  // Because LEAD/LAG.

    // For LEAD(0)/LAG(0), no computation need be performed.
    // Return copy of input.
    if (0 == static_cast<cudf::detail::lead_lag_aggregation*>(agg.get())->row_offset) {
      return std::make_unique<column>(input, stream, mr);
    }

    auto output = make_fixed_width_column(
      target_type(input.type(), op), input.size(), mask_state::UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();
    auto valid_count =
      kernel_launcher<T, agg_op, op, PrecedingWindowIterator, FollowingWindowIterator>(
        input,
        default_outputs,
        output_view,
        preceding_window_begin,
        following_window_begin,
        min_periods,
        agg,
        device_agg_op,
        stream);

    output->set_null_count(output->size() - valid_count);

    return output;
  }

  // Deals with invalid column and/or aggregation options
  template <typename T,
            typename agg_op,
            aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!(op == aggregation::LEAD || op == aggregation::LAG) ||
                     !cudf::is_fixed_width<T>(),
                   std::unique_ptr<column>>
  launch(column_view const& input,
         column_view const& default_outputs,
         PrecedingWindowIterator preceding_window_begin,
         FollowingWindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         agg_op device_agg_op,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL(
      "Aggregation operator and/or input type combination is invalid: "
      "LEAD/LAG supported only on fixed-width types");
  }

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!(op == aggregation::MEAN || op == aggregation::LEAD || op == aggregation::LAG),
                   std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& default_outputs,
             PrecedingWindowIterator preceding_window_begin,
             FollowingWindowIterator following_window_begin,
             size_type min_periods,
             std::unique_ptr<aggregation> const& agg,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    CUDF_EXPECTS(default_outputs.is_empty(),
                 "Only LEAD/LAG window functions support default values.");

    return launch<InputType,
                  typename corresponding_operator<op>::type,
                  op,
                  PrecedingWindowIterator,
                  FollowingWindowIterator>(input,
                                           default_outputs,
                                           preceding_window_begin,
                                           following_window_begin,
                                           min_periods,
                                           agg,
                                           stream,
                                           mr);
  }

  // This variant is just to handle mean
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<(op == aggregation::MEAN), std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& default_outputs,
    PrecedingWindowIterator preceding_window_begin,
    FollowingWindowIterator following_window_begin,
    size_type min_periods,
    std::unique_ptr<aggregation> const& agg,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    return launch<InputType, cudf::DeviceSum, op, PrecedingWindowIterator, FollowingWindowIterator>(
      input,
      default_outputs,
      preceding_window_begin,
      following_window_begin,
      min_periods,
      agg,
      stream,
      mr);
  }

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<(op == aggregation::LEAD || op == aggregation::LAG), std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& default_outputs,
             PrecedingWindowIterator preceding_window_begin,
             FollowingWindowIterator following_window_begin,
             size_type min_periods,
             std::unique_ptr<aggregation> const& agg,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    return launch<InputType,
                  cudf::DeviceLeadLag,
                  op,
                  PrecedingWindowIterator,
                  FollowingWindowIterator>(
      input,
      default_outputs,
      preceding_window_begin,
      following_window_begin,
      min_periods,
      agg,
      cudf::DeviceLeadLag{static_cast<cudf::detail::lead_lag_aggregation*>(agg.get())->row_offset},
      stream,
      mr);
  }
};

struct dispatch_rolling {
  template <typename T, typename PrecedingWindowIterator, typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     PrecedingWindowIterator preceding_window_begin,
                                     FollowingWindowIterator following_window_begin,
                                     size_type min_periods,
                                     std::unique_ptr<aggregation> const& agg,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return aggregation_dispatcher(agg->kind,
                                  rolling_window_launcher<T>{},
                                  input,
                                  default_outputs,
                                  preceding_window_begin,
                                  following_window_begin,
                                  min_periods,
                                  agg,
                                  stream,
                                  mr);
  }
};

}  // namespace

// Applies a user-defined rolling window function to the values in a column.
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window_udf(column_view const& input,
                                           PrecedingWindowIterator preceding_window,
                                           std::string const& preceding_window_str,
                                           FollowingWindowIterator following_window,
                                           std::string const& following_window_str,
                                           size_type min_periods,
                                           std::unique_ptr<aggregation> const& agg,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.has_nulls())
    CUDF_FAIL("Currently the UDF version of rolling window does NOT support inputs with nulls.");

  min_periods = std::max(min_periods, 0);

  auto udf_agg = static_cast<udf_aggregation*>(agg.get());

  std::string hash = "prog_rolling." + std::to_string(std::hash<std::string>{}(udf_agg->_source));

  std::string cuda_source;
  switch (udf_agg->kind) {
    case aggregation::Kind::PTX:
      cuda_source = cudf::rolling::jit::code::kernel_headers;
      cuda_source +=
        cudf::jit::parse_single_function_ptx(udf_agg->_source,
                                             udf_agg->_function_name,
                                             cudf::jit::get_type_name(udf_agg->_output_type),
                                             {0, 5});  // args 0 and 5 are pointers.
      cuda_source += cudf::rolling::jit::code::kernel;
      break;
    case aggregation::Kind::CUDA:
      cuda_source = cudf::rolling::jit::code::kernel_headers;
      cuda_source +=
        cudf::jit::parse_single_function_cuda(udf_agg->_source, udf_agg->_function_name);
      cuda_source += cudf::rolling::jit::code::kernel;
      break;
    default: CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(
    udf_agg->_output_type, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  rmm::device_scalar<size_type> device_valid_count{0, stream};

  const std::vector<std::string> compiler_flags{"-std=c++14",
                                                // Have jitify prune unused global variables
                                                "-remove-unused-globals",
                                                // suppress all NVRTC warnings
                                                "-w"};

  // Launch the jitify kernel
  cudf::jit::launcher(hash,
                      cuda_source,
                      {cudf_types_hpp,
                       cudf_utilities_bit_hpp,
                       cudf::rolling::jit::code::operation_h,
                       ___src_rolling_rolling_jit_detail_hpp},
                      compiler_flags,
                      nullptr,
                      stream)
    .set_kernel_inst("gpu_rolling_new",  // name of the kernel we are launching
                     {cudf::jit::get_type_name(input.type()),  // list of template arguments
                      cudf::jit::get_type_name(output->type()),
                      udf_agg->_operator_name,
                      preceding_window_str.c_str(),
                      following_window_str.c_str()})
    .launch(input.size(),
            cudf::jit::get_data_ptr(input),
            input.null_mask(),
            cudf::jit::get_data_ptr(output_view),
            output_view.null_mask(),
            device_valid_count.data(),
            preceding_window,
            following_window,
            min_periods);

  output->set_null_count(output->size() - device_valid_count.value(stream));

  // check the stream for debugging
  CHECK_CUDA(stream.value());

  return output;
}

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               PrecedingWindowIterator preceding_window_begin,
 *                               FollowingWindowIterator following_window_begin,
 *                               size_type min_periods,
 *                               std::unique_ptr<aggregation> const& agg,
 *                               rmm::mr::device_memory_resource* mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       PrecedingWindowIterator preceding_window_begin,
                                       FollowingWindowIterator following_window_begin,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  min_periods = std::max(min_periods, 0);

  return cudf::type_dispatcher(input.type(),
                               dispatch_rolling{},
                               input,
                               default_outputs,
                               preceding_window_begin,
                               following_window_begin,
                               min_periods,
                               agg,
                               stream,
                               mr);
}

}  // namespace detail

}  // namespace cudf
