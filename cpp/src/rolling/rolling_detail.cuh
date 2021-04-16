/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "rolling_detail.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/rolling.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <jit/cache.hpp>
#include <jit/parser.hpp>
#include <jit/type.hpp>

#include <jit_preprocessed_files/rolling/jit/kernel.cu.jit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/binary_search.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <memory>
#include "lead_lag_nested_detail.cuh"

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
                             op == aggregation::LAG || op == aggregation::COLLECT_LIST)>* = nullptr>
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
         PrecedingWindowIterator preceding,
         FollowingWindowIterator following,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         agg_op const& device_agg_op,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr)
  {
    auto output = make_fixed_width_column(
      target_type(input.type(), op), input.size(), mask_state::UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();
    auto valid_count =
      kernel_launcher<T, agg_op, op, PrecedingWindowIterator, FollowingWindowIterator>(
        input,
        default_outputs,
        output_view,
        preceding,
        following,
        min_periods,
        agg,
        device_agg_op,
        stream);

    output->set_null_count(output->size() - valid_count);

    return output;
  }

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!(op == aggregation::MEAN || op == aggregation::LEAD || op == aggregation::LAG ||
                     op == aggregation::COLLECT_LIST),
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
  std::enable_if_t<cudf::is_fixed_width<InputType>() &&
                     (op == aggregation::LEAD || op == aggregation::LAG),
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

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!cudf::is_fixed_width<InputType>() &&
                     (op == aggregation::LEAD || op == aggregation::LAG),
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
    return cudf::detail::
      compute_lead_lag_for_nested<op, InputType, PrecedingWindowIterator, FollowingWindowIterator>(
        input,
        default_outputs,
        preceding_window_begin,
        following_window_begin,
        static_cast<cudf::detail::lead_lag_aggregation*>(agg.get())->row_offset,
        stream,
        mr);
  }

  /**
   * @brief Creates the offsets child of the result of the `COLLECT_LIST` window aggregation
   *
   * Given the input column, the preceding/following window bounds, and `min_periods`,
   * the sizes of each list row may be computed. These values can then be used to
   * calculate the offsets for the result of `COLLECT_LIST`.
   *
   * Note: If `min_periods` exceeds the number of observations for a window, the size
   * is set to `0` (since the result is `null`).
   */
  template <typename PrecedingIter, typename FollowingIter>
  std::unique_ptr<column> create_collect_offsets(size_type input_size,
                                                 PrecedingIter preceding_begin,
                                                 FollowingIter following_begin,
                                                 size_type min_periods,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
  {
    // Materialize offsets column.
    auto static constexpr size_data_type = data_type{type_to_id<size_type>()};
    auto sizes =
      make_fixed_width_column(size_data_type, input_size, mask_state::UNALLOCATED, stream, mr);
    auto mutable_sizes = sizes->mutable_view();

    // Consider the following preceding/following values:
    //    preceding = [1,2,2,2,2]
    //    following = [1,1,1,1,0]
    // The sum of the vectors should yield the window sizes:
    //  prec + foll = [2,3,3,3,2]
    //
    // If min_periods=2, all rows have at least `min_periods` observations.
    // But if min_periods=3, rows at indices 0 and 4 have too few observations, and must return
    // null. The sizes at these positions must be 0, i.e.
    //  prec + foll = [0,3,3,3,0]
    thrust::transform(rmm::exec_policy(stream),
                      preceding_begin,
                      preceding_begin + input_size,
                      following_begin,
                      mutable_sizes.begin<size_type>(),
                      [min_periods] __device__(auto preceding, auto following) {
                        return (preceding + following) < min_periods ? 0 : (preceding + following);
                      });

    // Convert `sizes` to an offsets column, via inclusive_scan():
    return strings::detail::make_offsets_child_column(
      sizes->view().begin<size_type>(), sizes->view().end<size_type>(), stream, mr);
  }

  /**
   * @brief Generate mapping of each row in the COLLECT_LIST result's child column
   * to the index of the row it belongs to.
   *
   *  If
   *         input col == [A,B,C,D,E]
   *    and  preceding == [1,2,2,2,2],
   *    and  following == [1,1,1,1,0],
   *  then,
   *        collect result       == [ [A,B], [A,B,C], [B,C,D], [C,D,E], [D,E] ]
   *   i.e. result offset column == [0,2,5,8,11,13],
   *    and result child  column == [A,B,A,B,C,B,C,D,C,D,E,D,E].
   *  Mapping back to `input`    == [0,1,0,1,2,1,2,3,2,3,4,3,4]
   */
  std::unique_ptr<column> get_list_child_to_list_row_mapping(cudf::column_view const& offsets,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::mr::device_memory_resource* mr)
  {
    auto static constexpr size_data_type = data_type{type_to_id<size_type>()};

    // First, reduce offsets column by key, to identify the number of times
    // an offset appears.
    // Next, scatter the count for each offset (except the first and last),
    // into a column of N `0`s, where N == number of child rows.
    // For the example above:
    //   offsets        == [0, 2, 5, 8, 11, 13]
    //   scatter result == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    //
    // If the above example had an empty list row at index 2,
    // the same columns would look as follows:
    //   offsets        == [0, 2, 5, 5, 8, 11, 13]
    //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0]
    //
    // Note: To correctly handle null list rows at the beginning of
    // the output column, care must be taken to skip the first `0`
    // in the offsets column, when running `reduce_by_key()`.
    // This accounts for the `0` added by default to the offsets
    // column, marking the beginning of the column.

    auto const num_child_rows{
      cudf::detail::get_value<size_type>(offsets, offsets.size() - 1, stream)};

    auto scatter_values =
      make_fixed_width_column(size_data_type, offsets.size(), mask_state::UNALLOCATED, stream, mr);
    auto scatter_keys =
      make_fixed_width_column(size_data_type, offsets.size(), mask_state::UNALLOCATED, stream, mr);
    auto reduced_by_key =
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            offsets.template begin<size_type>() + 1,  // Skip first 0 in offsets.
                            offsets.template end<size_type>(),
                            thrust::make_constant_iterator<size_type>(1),
                            scatter_keys->mutable_view().template begin<size_type>(),
                            scatter_values->mutable_view().template begin<size_type>());
    auto scatter_values_end = reduced_by_key.second;
    auto scatter_output =
      make_fixed_width_column(size_data_type, num_child_rows, mask_state::UNALLOCATED, stream, mr);
    thrust::fill_n(rmm::exec_policy(stream),
                   scatter_output->mutable_view().template begin<size_type>(),
                   num_child_rows,
                   0);  // [0,0,0,...0]
    thrust::scatter(
      rmm::exec_policy(stream),
      scatter_values->mutable_view().template begin<size_type>(),
      scatter_values_end,
      scatter_keys->view().template begin<size_type>(),
      scatter_output->mutable_view().template begin<size_type>());  // [0,0,1,0,0,1,...]

    // Next, generate mapping with inclusive_scan() on scatter() result.
    // For the example above:
    //   scatter result == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    //   inclusive_scan == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    //
    // For the case with an empty list at index 3:
    //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0]
    //   inclusive_scan == [0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5]
    auto per_row_mapping =
      make_fixed_width_column(size_data_type, num_child_rows, mask_state::UNALLOCATED, stream, mr);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           scatter_output->view().template begin<size_type>(),
                           scatter_output->view().template end<size_type>(),
                           per_row_mapping->mutable_view().template begin<size_type>());
    return per_row_mapping;
  }

  /**
   * @brief Create gather map to generate the child column of the result of
   * the `COLLECT_LIST` window aggregation.
   */
  template <typename PrecedingIter>
  std::unique_ptr<column> create_collect_gather_map(column_view const& child_offsets,
                                                    column_view const& per_row_mapping,
                                                    PrecedingIter preceding_iter,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
  {
    auto gather_map = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                              per_row_mapping.size(),
                                              mask_state::UNALLOCATED,
                                              stream,
                                              mr);
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(per_row_mapping.size()),
      gather_map->mutable_view().template begin<size_type>(),
      [d_offsets =
         child_offsets.template begin<size_type>(),  // E.g. [0,   2,     5,     8,     11, 13]
       d_groups =
         per_row_mapping.template begin<size_type>(),  // E.g. [0,0, 1,1,1, 2,2,2, 3,3,3, 4,4]
       d_prev = preceding_iter] __device__(auto i) {
        auto group              = d_groups[i];
        auto group_start_offset = d_offsets[group];
        auto relative_index     = i - group_start_offset;

        return (group - d_prev[group] + 1) + relative_index;
      });
    return gather_map;
  }

  /**
   * @brief Count null entries in result of COLLECT_LIST.
   */
  size_type count_child_nulls(column_view const& input,
                              std::unique_ptr<column> const& gather_map,
                              rmm::cuda_stream_view stream)
  {
    auto input_device_view = column_device_view::create(input, stream);

    auto input_row_is_null = [d_input = *input_device_view] __device__(auto i) {
      return d_input.is_null_nocheck(i);
    };

    return thrust::count_if(rmm::exec_policy(stream),
                            gather_map->view().template begin<size_type>(),
                            gather_map->view().template end<size_type>(),
                            input_row_is_null);
  }

  /**
   * @brief Purge entries for null inputs from gather_map, and adjust offsets.
   */
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
    column_view const& input,
    column_view const& gather_map,
    column_view const& offsets,
    size_type num_child_nulls,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto input_device_view = column_device_view::create(input, stream);

    auto input_row_not_null = [d_input = *input_device_view] __device__(auto i) {
      return d_input.is_valid_nocheck(i);
    };

    // Purge entries in gather_map that correspond to null input.
    auto new_gather_map = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                                  gather_map.size() - num_child_nulls,
                                                  mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
    thrust::copy_if(rmm::exec_policy(stream),
                    gather_map.template begin<size_type>(),
                    gather_map.template end<size_type>(),
                    new_gather_map->mutable_view().template begin<size_type>(),
                    input_row_not_null);

    // Recalculate offsets after null entries are purged.
    auto new_sizes = make_fixed_width_column(
      data_type{type_to_id<size_type>()}, input.size(), mask_state::UNALLOCATED, stream, mr);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      new_sizes->mutable_view().template begin<size_type>(),
                      [d_gather_map  = gather_map.template begin<size_type>(),
                       d_old_offsets = offsets.template begin<size_type>(),
                       input_row_not_null] __device__(auto i) {
                        return thrust::count_if(thrust::seq,
                                                d_gather_map + d_old_offsets[i],
                                                d_gather_map + d_old_offsets[i + 1],
                                                input_row_not_null);
                      });

    auto new_offsets =
      strings::detail::make_offsets_child_column(new_sizes->view().template begin<size_type>(),
                                                 new_sizes->view().template end<size_type>(),
                                                 stream,
                                                 mr);

    return std::make_pair<std::unique_ptr<column>, std::unique_ptr<column>>(
      std::move(new_gather_map), std::move(new_offsets));
  }

  template <aggregation::Kind op, typename PrecedingIter, typename FollowingIter>
  std::enable_if_t<(op == aggregation::COLLECT_LIST), std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& default_outputs,
    PrecedingIter preceding_begin_raw,
    FollowingIter following_begin_raw,
    size_type min_periods,
    std::unique_ptr<aggregation> const& agg,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUDF_EXPECTS(default_outputs.is_empty(),
                 "COLLECT_LIST window function does not support default values.");

    if (input.is_empty()) return empty_like(input);

    // Fix up preceding/following iterators to respect column boundaries,
    // similar to gpu_rolling().
    // `rolling_window()` does not fix up preceding/following so as not to read past
    // column boundaries.
    // `grouped_rolling_window()` and `time_range_based_grouped_rolling_window() do.
    auto preceding_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0), [preceding_begin_raw] __device__(auto i) {
        return thrust::min(preceding_begin_raw[i], i + 1);
      });
    auto following_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      [following_begin_raw, size = input.size()] __device__(auto i) {
        return thrust::min(following_begin_raw[i], size - i - 1);
      });

    // Materialize collect list's offsets.
    auto offsets = create_collect_offsets(
      input.size(), preceding_begin, following_begin, min_periods, stream, mr);

    // Map each element of the collect() result's child column
    // to the index where it appears in the input.
    auto per_row_mapping = get_list_child_to_list_row_mapping(offsets->view(), stream, mr);

    // Generate gather map to produce the collect() result's child column.
    auto gather_map = create_collect_gather_map(
      offsets->view(), per_row_mapping->view(), preceding_begin, stream, mr);

    // If gather_map collects null elements, and null_policy == EXCLUDE,
    // those elements must be filtered out, and offsets recomputed.
    auto null_handling = static_cast<collect_list_aggregation*>(agg.get())->_null_handling;
    if (null_handling == null_policy::EXCLUDE && input.has_nulls()) {
      auto num_child_nulls = count_child_nulls(input, gather_map, stream);
      if (num_child_nulls != 0) {
        std::tie(gather_map, offsets) =
          purge_null_entries(input, *gather_map, *offsets, num_child_nulls, stream, mr);
      }
    }

    // gather(), to construct child column.
    auto gather_output =
      cudf::gather(table_view{std::vector<column_view>{input}}, gather_map->view());

    rmm::device_buffer null_mask;
    size_type null_count;
    std::tie(null_mask, null_count) = valid_if(
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(input.size()),
      [preceding_begin, following_begin, min_periods] __device__(auto i) {
        return (preceding_begin[i] + following_begin[i]) >= min_periods;
      },
      stream,
      mr);

    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(gather_output->release()[0]),
                             null_count,
                             std::move(null_mask),
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
      cuda_source +=
        cudf::jit::parse_single_function_ptx(udf_agg->_source,
                                             udf_agg->_function_name,
                                             cudf::jit::get_type_name(udf_agg->_output_type),
                                             {0, 5});  // args 0 and 5 are pointers.
      break;
    case aggregation::Kind::CUDA:
      cuda_source +=
        cudf::jit::parse_single_function_cuda(udf_agg->_source, udf_agg->_function_name);
      break;
    default: CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(
    udf_agg->_output_type, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  rmm::device_scalar<size_type> device_valid_count{0, stream};

  std::string kernel_name =
    jitify2::reflection::Template("cudf::rolling::jit::gpu_rolling_new")  //
      .instantiate(cudf::jit::get_type_name(input.type()),  // list of template arguments
                   cudf::jit::get_type_name(output->type()),
                   udf_agg->_operator_name,
                   preceding_window_str.c_str(),
                   following_window_str.c_str());

  cudf::jit::get_program_cache(*rolling_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"rolling/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                 //
    ->launch(input.size(),
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

  if (input.is_empty()) return empty_like(input);

  if (cudf::is_dictionary(input.type()))
    CUDF_EXPECTS(agg->kind == aggregation::COUNT_ALL || agg->kind == aggregation::COUNT_VALID ||
                   agg->kind == aggregation::ROW_NUMBER || agg->kind == aggregation::MIN ||
                   agg->kind == aggregation::MAX || agg->kind == aggregation::LEAD ||
                   agg->kind == aggregation::LAG,
                 "Invalid aggregation for dictionary column");

  min_periods = std::max(min_periods, 0);

  auto input_col = cudf::is_dictionary(input.type())
                     ? dictionary_column_view(input).get_indices_annotated()
                     : input;

  auto output = cudf::type_dispatcher(input_col.type(),
                                      dispatch_rolling{},
                                      input_col,
                                      default_outputs,
                                      preceding_window_begin,
                                      following_window_begin,
                                      min_periods,
                                      agg,
                                      stream,
                                      mr);
  if (!cudf::is_dictionary(input.type())) return output;

  // dictionary column post processing
  if (agg->kind == aggregation::COUNT_ALL || agg->kind == aggregation::COUNT_VALID ||
      agg->kind == aggregation::ROW_NUMBER)
    return output;

  // output is new dictionary indices (including nulls)
  auto keys = std::make_unique<column>(dictionary_column_view(input).keys(), stream, mr);
  auto const indices_type = output->type();        // capture these
  auto const output_size  = output->size();        // before calling
  auto const null_count   = output->null_count();  // release()
  auto contents           = output->release();
  // create indices column from output column data
  auto indices = std::make_unique<column>(indices_type,
                                          output_size,
                                          std::move(*(contents.data.release())),
                                          rmm::device_buffer{0, stream, mr},
                                          0);
  // create dictionary from keys and indices
  return make_dictionary_column(
    std::move(keys), std::move(indices), std::move(*(contents.null_mask.release())), null_count);
}

}  // namespace detail

}  // namespace cudf
