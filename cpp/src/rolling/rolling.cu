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

  bool output_is_valid                      = (count >= min_periods);
  output.element<OutputType>(current_index) = count;

  return output_is_valid;
}

/**
 * @brief Calculates row-number within [start_index, end_index).
 *        Count is updated depending on `min_periods`
 *        Returns true if it was valid, else false.
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
  bool output_is_valid                      = ((end_index - start_index) >= min_periods);
  output.element<OutputType>(current_index) = ((current_index - start_index) + 1);

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
    constexpr cudf::size_type block_size = 256;
    cudf::detail::grid_1d grid(input.size(), block_size);

    auto input_device_view           = column_device_view::create(input, stream);
    auto output_device_view          = mutable_column_device_view::create(output, stream);
    auto default_outputs_device_view = column_device_view::create(default_outputs, stream);

    rmm::device_scalar<size_type> device_valid_count{0, stream};

    if (input.has_nulls()) {
      gpu_rolling<T, target_type_t<InputType, op>, agg_op, op, block_size, true>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods);
    } else {
      gpu_rolling<T, target_type_t<InputType, op>, agg_op, op, block_size, false>
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
    constexpr cudf::size_type block_size = 256;
    cudf::detail::grid_1d grid(input.size(), block_size);

    auto input_device_view           = column_device_view::create(input, stream);
    auto output_device_view          = mutable_column_device_view::create(output, stream);
    auto default_outputs_device_view = column_device_view::create(default_outputs, stream);

    rmm::device_scalar<size_type> device_valid_count{0, stream};

    if (input.has_nulls()) {
      gpu_rolling<T, target_type_t<InputType, op>, agg_op, op, block_size, true>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                             *default_outputs_device_view,
                                                             *output_device_view,
                                                             device_valid_count.data(),
                                                             preceding_window_begin,
                                                             following_window_begin,
                                                             min_periods,
                                                             device_agg_op);
    } else {
      gpu_rolling<T, target_type_t<InputType, op>, agg_op, op, block_size, false>
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
                                       detail::out_of_bounds_policy::IGNORE,
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

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  return rolling_window(
    input, empty_like(input)->view(), preceding_window, following_window, min_periods, agg, mr);
}

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) return empty_like(input);
  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  if (agg->kind == aggregation::CUDA || agg->kind == aggregation::PTX) {
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window,
                                            "cudf::size_type",
                                            following_window,
                                            "cudf::size_type",
                                            min_periods,
                                            agg,
                                            rmm::cuda_stream_default,
                                            mr);
  } else {
    auto preceding_window_begin = thrust::make_constant_iterator(preceding_window);
    auto following_window_begin = thrust::make_constant_iterator(following_window);

    return cudf::detail::rolling_window(input,
                                        default_outputs,
                                        preceding_window_begin,
                                        following_window_begin,
                                        min_periods,
                                        agg,
                                        rmm::cuda_stream_default,
                                        mr);
  }
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (preceding_window.is_empty() || following_window.is_empty() || input.is_empty())
    return empty_like(input);

  CUDF_EXPECTS(preceding_window.type().id() == type_id::INT32 &&
                 following_window.type().id() == type_id::INT32,
               "preceding_window/following_window must have type_id::INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  if (agg->kind == aggregation::CUDA || agg->kind == aggregation::PTX) {
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window.begin<size_type>(),
                                            "cudf::size_type*",
                                            following_window.begin<size_type>(),
                                            "cudf::size_type*",
                                            min_periods,
                                            agg,
                                            rmm::cuda_stream_default,
                                            mr);
  } else {
    return cudf::detail::rolling_window(input,
                                        empty_like(input)->view(),
                                        preceding_window.begin<size_type>(),
                                        following_window.begin<size_type>(),
                                        min_periods,
                                        agg,
                                        rmm::cuda_stream_default,
                                        mr);
  }
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               window_bounds preceding_window,
                                               window_bounds following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                empty_like(input)->view(),
                                preceding_window,
                                following_window,
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                default_outputs,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) return empty_like(input);

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  auto const preceding_window = preceding_window_bounds.value;
  auto const following_window = following_window_bounds.value;

  if (group_keys.num_columns() == 0) {
    // No Groupby columns specified. Treat as one big group.
    return rolling_window(
      input, default_outputs, preceding_window, following_window, min_periods, aggr, mr);
  }

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
  auto group_offsets{helper.group_offsets()};
  auto const& group_labels{helper.group_labels()};

  // `group_offsets` are interpreted in adjacent pairs, each pair representing the offsets
  // of the first, and one past the last elements in a group.
  //
  // If `group_offsets` is not empty, it must contain at least two offsets:
  //   a. 0, indicating the first element in `input`
  //   b. input.size(), indicating one past the last element in `input`.
  //
  // Thus, for an input of 1000 rows,
  //   0. [] indicates a single group, spanning the entire column.
  //   1  [10] is invalid.
  //   2. [0, 1000] indicates a single group, spanning the entire column (thus, equivalent to no
  //   groups.)
  //   3. [0, 500, 1000] indicates two equal-sized groups: [0,500), and [500,1000).

  assert(group_offsets.size() >= 2 && group_offsets[0] == 0 &&
         group_offsets[group_offsets.size() - 1] == input.size() &&
         "Must have at least one group.");

  auto preceding_calculator = [d_group_offsets = group_offsets.data().get(),
                               d_group_labels  = group_labels.data().get(),
                               preceding_window] __device__(size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    return thrust::minimum<size_type>{}(preceding_window,
                                        idx - group_start + 1);  // Preceding includes current row.
  };

  auto following_calculator = [d_group_offsets = group_offsets.data().get(),
                               d_group_labels  = group_labels.data().get(),
                               following_window] __device__(size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_end =
      d_group_offsets[group_label +
                      1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    return thrust::minimum<size_type>{}(following_window, (group_end - 1) - idx);
  };

  if (aggr->kind == aggregation::CUDA || aggr->kind == aggregation::PTX) {
    cudf::detail::preceding_window_wrapper grouped_preceding_window{
      group_offsets.data().get(), group_labels.data().get(), preceding_window};

    cudf::detail::following_window_wrapper grouped_following_window{
      group_offsets.data().get(), group_labels.data().get(), following_window};

    return cudf::detail::rolling_window_udf(input,
                                            grouped_preceding_window,
                                            "cudf::detail::preceding_window_wrapper",
                                            grouped_following_window,
                                            "cudf::detail::following_window_wrapper",
                                            min_periods,
                                            aggr,
                                            rmm::cuda_stream_default,
                                            mr);
  } else {
    return cudf::detail::rolling_window(
      input,
      default_outputs,
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      preceding_calculator),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      following_calculator),
      min_periods,
      aggr,
      rmm::cuda_stream_default,
      mr);
  }
}

namespace {

bool is_supported_range_frame_unit(cudf::data_type const& data_type)
{
  auto id = data_type.id();
  return id == cudf::type_id::TIMESTAMP_DAYS || id == cudf::type_id::TIMESTAMP_SECONDS ||
         id == cudf::type_id::TIMESTAMP_MILLISECONDS ||
         id == cudf::type_id::TIMESTAMP_MICROSECONDS || id == cudf::type_id::TIMESTAMP_NANOSECONDS;
}

/// Fetches multiplication factor to normalize window sizes, depending on the datatype of the
/// timestamp column. Used for time-based rolling-window operations. E.g. If the timestamp column is
/// in TIMESTAMP_SECONDS, and the window sizes are specified in DAYS, the window size needs to be
/// multiplied by `24*60*60`, before comparisons with the timestamps.
size_t multiplication_factor(cudf::data_type const& data_type)
{
  // Assume timestamps.
  switch (data_type.id()) {
    case cudf::type_id::TIMESTAMP_DAYS: return 1L;
    case cudf::type_id::TIMESTAMP_SECONDS: return 24L * 60 * 60;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 24L * 60 * 60 * 1000;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 24L * 60 * 60 * 1000 * 1000;
    default:
      CUDF_EXPECTS(data_type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS,
                   "Unexpected data-type for timestamp-based rolling window operation!");
      return 24L * 60 * 60 * 1000 * 1000 * 1000;
  }
}

/// Given a single, ungrouped timestamp column, return the indices corresponding
/// to the first null timestamp, and (one past) the last null timestamp.
/// The input column is sorted, with all null values clustered either
/// at the beginning of the column or at the end.
/// If no null values are founds, null_begin and null_end are 0.
std::tuple<size_type, size_type> get_null_bounds_for_timestamp_column(
  column_view const& timestamp_column)
{
  auto const num_rows  = timestamp_column.size();
  auto const num_nulls = timestamp_column.null_count();

  if (num_nulls == num_rows || num_nulls == 0) {
    // Short-circuit: All nulls, or no nulls.
    return std::make_tuple(0, num_nulls);
  }

  auto const first_row_is_null = timestamp_column.null_count(0, 1) == 1;

  return first_row_is_null ? std::make_tuple(0, num_nulls)
                           : std::make_tuple(num_rows - num_nulls, num_rows);
}

/// Time-range window computation, with
///   1. no grouping keys specified
///   2. timetamps in ASCENDING order.
/// Treat as one single group.
template <typename TimeT>
std::unique_ptr<column> time_range_window_ASC(column_view const& input,
                                              column_view const& timestamp_column,
                                              TimeT preceding_window,
                                              bool preceding_window_is_unbounded,
                                              TimeT following_window,
                                              bool following_window_is_unbounded,
                                              size_type min_periods,
                                              std::unique_ptr<aggregation> const& aggr,
                                              rmm::mr::device_memory_resource* mr)
{
  size_type nulls_begin_idx, nulls_end_idx;
  std::tie(nulls_begin_idx, nulls_end_idx) = get_null_bounds_for_timestamp_column(timestamp_column);

  auto preceding_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     d_timestamps = timestamp_column.data<TimeT>(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (preceding_window_is_unbounded) {
      return idx + 1;  // Technically `idx - 0 + 1`,
                       // where 0 == Group start,
                       // and   1 accounts for the current row
    }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Must consider beginning of null-group as window start.
      return idx - nulls_begin_idx + 1;
    }

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
    //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
    // Otherwise, NULLS LAST ordering. Start at 0.
    auto group_start                = nulls_begin_idx == 0 ? nulls_end_idx : 0;
    auto lowest_timestamp_in_window = d_timestamps[idx] - preceding_window;

    return ((d_timestamps + idx) - thrust::lower_bound(thrust::seq,
                                                       d_timestamps + group_start,
                                                       d_timestamps + idx,
                                                       lowest_timestamp_in_window)) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     num_rows     = input.size(),
     d_timestamps = timestamp_column.data<TimeT>(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (following_window_is_unbounded) { return num_rows - idx - 1; }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Window ends at the end of the null group.
      return nulls_end_idx - idx - 1;
    }

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search ends at num_rows.
    //  2. NO NULLS: Binary search also ends at num_rows.
    // Otherwise, NULLS LAST ordering. End at nulls_begin_idx.

    auto group_end                   = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
    auto highest_timestamp_in_window = d_timestamps[idx] + following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + group_end,
                                highest_timestamp_in_window) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

/// Given a timestamp column grouped as specified in group_offsets,
/// return the following two vectors:
///  1. Vector with one entry per group, indicating the offset in the group
///     where the null values begin.
///  2. Vector with one entry per group, indicating the offset in the group
///     where the null values end. (i.e. 1 past the last null.)
/// Each group in the input timestamp column must be sorted,
/// with null values clustered at either the start or the end of each group.
/// If there are no nulls for any given group, (nulls_begin, nulls_end) == (0,0).
std::tuple<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_null_bounds_for_timestamp_column(column_view const& timestamp_column,
                                     rmm::device_vector<size_type> const& group_offsets)
{
  // For each group, the null values are themselves clustered
  // at the beginning or the end of the group.
  // These nulls cannot participate, except in their own window.

  // If the input has n groups, group_offsets will have n+1 values.
  // null_start and null_end should eventually have 1 entry per group.
  auto null_start = rmm::device_vector<size_type>(group_offsets.begin(), group_offsets.end() - 1);
  auto null_end   = rmm::device_vector<size_type>(group_offsets.begin(), group_offsets.end() - 1);

  if (timestamp_column.has_nulls()) {
    auto p_timestamps_device_view = column_device_view::create(timestamp_column);
    auto num_groups               = group_offsets.size();

    // Null timestamps exist. Find null bounds, per group.
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator(static_cast<size_type>(0)),
      thrust::make_counting_iterator(static_cast<size_type>(num_groups)),
      [d_timestamps    = *p_timestamps_device_view,
       d_group_offsets = group_offsets.data().get(),
       d_null_start    = null_start.data(),
       d_null_end      = null_end.data()] __device__(auto group_label) {
        auto group_start           = d_group_offsets[group_label];
        auto group_end             = d_group_offsets[group_label + 1];
        auto first_element_is_null = d_timestamps.is_null_nocheck(group_start);
        auto last_element_is_null  = d_timestamps.is_null_nocheck(group_end - 1);
        if (!first_element_is_null && !last_element_is_null) {
          // Short circuit: No nulls.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = group_start;
        } else if (first_element_is_null && last_element_is_null) {
          // Short circuit: All nulls.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = group_end;
        } else if (first_element_is_null) {
          // NULLS FIRST.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = *thrust::partition_point(
            thrust::seq,
            thrust::make_counting_iterator(group_start),
            thrust::make_counting_iterator(group_end),
            [&d_timestamps] __device__(auto i) { return d_timestamps.is_null_nocheck(i); });
        } else {
          // NULLS LAST.
          d_null_end[group_label]   = group_end;
          d_null_start[group_label] = *thrust::partition_point(
            thrust::seq,
            thrust::make_counting_iterator(group_start),
            thrust::make_counting_iterator(group_end),
            [&d_timestamps] __device__(auto i) { return d_timestamps.is_valid_nocheck(i); });
        }
      });
  }

  return std::make_tuple(std::move(null_start), std::move(null_end));
}

// Time-range window computation, for timestamps in ASCENDING order.
template <typename TimeT>
std::unique_ptr<column> time_range_window_ASC(
  column_view const& input,
  column_view const& timestamp_column,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  TimeT preceding_window,
  bool preceding_window_is_unbounded,
  TimeT following_window,
  bool following_window_is_unbounded,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  rmm::device_vector<size_type> null_start, null_end;
  std::tie(null_start, null_end) =
    get_null_bounds_for_timestamp_column(timestamp_column, group_offsets);

  auto preceding_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (preceding_window_is_unbounded) { return idx - group_start + 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window starts at the start of the null group.
      return idx - nulls_begin + 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search must begin at nulls_end.
    //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
    // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
    auto search_start = nulls_begin == group_start ? nulls_end : group_start;

    auto lowest_timestamp_in_window = d_timestamps[idx] - preceding_window;

    return ((d_timestamps + idx) - thrust::lower_bound(thrust::seq,
                                                       d_timestamps + search_start,
                                                       d_timestamps + idx,
                                                       lowest_timestamp_in_window)) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto group_end =
      d_group_offsets[group_label +
                      1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (following_window_is_unbounded) { return (group_end - idx) - 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window ends at the end of the null group.
      return nulls_end - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search ends at group_end.
    //  2. NO NULLS: Search ends at group_end.
    // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
    auto search_end = nulls_begin == group_start ? group_end : nulls_begin;

    auto highest_timestamp_in_window = d_timestamps[idx] + following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + search_end,
                                highest_timestamp_in_window) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

/// Time-range window computation, with
///   1. no grouping keys specified
///   2. timetamps in DESCENDING order.
/// Treat as one single group.
template <typename TimeT>
std::unique_ptr<column> time_range_window_DESC(column_view const& input,
                                               column_view const& timestamp_column,
                                               TimeT preceding_window,
                                               bool preceding_window_is_unbounded,
                                               TimeT following_window,
                                               bool following_window_is_unbounded,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  size_type nulls_begin_idx, nulls_end_idx;
  std::tie(nulls_begin_idx, nulls_end_idx) = get_null_bounds_for_timestamp_column(timestamp_column);

  auto preceding_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     d_timestamps = timestamp_column.data<TimeT>(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (preceding_window_is_unbounded) {
      return idx + 1;  // Technically `idx - 0 + 1`,
                       // where 0 == Group start,
                       // and   1 accounts for the current row
    }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Must consider beginning of null-group as window start.
      return idx - nulls_begin_idx + 1;
    }

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
    //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
    // Otherwise, NULLS LAST ordering. Start at 0.
    auto group_start                 = nulls_begin_idx == 0 ? nulls_end_idx : 0;
    auto highest_timestamp_in_window = d_timestamps[idx] + preceding_window;

    return ((d_timestamps + idx) -
            thrust::lower_bound(thrust::seq,
                                d_timestamps + group_start,
                                d_timestamps + idx,
                                highest_timestamp_in_window,
                                thrust::greater<decltype(highest_timestamp_in_window)>())) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     num_rows     = input.size(),
     d_timestamps = timestamp_column.data<TimeT>(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (following_window_is_unbounded) { return (num_rows - idx) - 1; }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Window ends at the end of the null group.
      return nulls_end_idx - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude null group.
    // If nulls_begin_idx = 0, either
    //  1. NULLS FIRST ordering: Search ends at num_rows.
    //  2. NO NULLS: Search also ends at num_rows.
    // Otherwise, NULLS LAST ordering: End at nulls_begin_idx.

    auto group_end                  = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
    auto lowest_timestamp_in_window = d_timestamps[idx] - following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + group_end,
                                lowest_timestamp_in_window,
                                thrust::greater<decltype(lowest_timestamp_in_window)>()) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

// Time-range window computation, for timestamps in DESCENDING order.
template <typename TimeT>
std::unique_ptr<column> time_range_window_DESC(
  column_view const& input,
  column_view const& timestamp_column,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  TimeT preceding_window,
  bool preceding_window_is_unbounded,
  TimeT following_window,
  bool following_window_is_unbounded,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  rmm::device_vector<size_type> null_start, null_end;
  std::tie(null_start, null_end) =
    get_null_bounds_for_timestamp_column(timestamp_column, group_offsets);

  auto preceding_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (preceding_window_is_unbounded) { return (idx - group_start) + 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window starts at the start of the null group.
      return idx - nulls_begin + 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search must begin at nulls_end.
    //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
    // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
    auto search_start = nulls_begin == group_start ? nulls_end : group_start;

    auto highest_timestamp_in_window = d_timestamps[idx] + preceding_window;

    return ((d_timestamps + idx) -
            thrust::lower_bound(thrust::seq,
                                d_timestamps + search_start,
                                d_timestamps + idx,
                                highest_timestamp_in_window,
                                thrust::greater<decltype(highest_timestamp_in_window)>())) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto group_end   = d_group_offsets[group_label + 1];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (following_window_is_unbounded) { return (group_end - idx) - 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window ends at the end of the null group.
      return nulls_end - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search ends at group_end.
    //  2. NO NULLS: Search ends at group_end.
    // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
    auto search_end = nulls_begin == group_start ? group_end : nulls_begin;

    auto lowest_timestamp_in_window = d_timestamps[idx] - following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + search_end,
                                lowest_timestamp_in_window,
                                thrust::greater<decltype(lowest_timestamp_in_window)>()) -
            (d_timestamps + idx)) -
           1;
  };

  if (aggr->kind == aggregation::CUDA || aggr->kind == aggregation::PTX) {
    CUDF_FAIL("Time ranged rolling window does NOT (yet) support UDF.");
  } else {
    return cudf::detail::rolling_window(
      input,
      empty_like(input)->view(),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      preceding_calculator),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      following_calculator),
      min_periods,
      aggr,
      rmm::cuda_stream_default,
      mr);
  }
}

template <typename TimeT>
std::unique_ptr<column> grouped_time_range_rolling_window_impl(
  column_view const& input,
  column_view const& timestamp_column,
  cudf::order const& timestamp_ordering,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  window_bounds preceding_window_in_days,  // TODO: Consider taking offset-type as type_id. Assumes
                                           // days for now.
  window_bounds following_window_in_days,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  TimeT mult_factor{static_cast<TimeT>(multiplication_factor(timestamp_column.type()))};

  if (timestamp_ordering == cudf::order::ASCENDING) {
    return group_offsets.empty()
             ? time_range_window_ASC(input,
                                     timestamp_column,
                                     preceding_window_in_days.value * mult_factor,
                                     preceding_window_in_days.is_unbounded,
                                     following_window_in_days.value * mult_factor,
                                     following_window_in_days.is_unbounded,
                                     min_periods,
                                     aggr,
                                     mr)
             : time_range_window_ASC(input,
                                     timestamp_column,
                                     group_offsets,
                                     group_labels,
                                     preceding_window_in_days.value * mult_factor,
                                     preceding_window_in_days.is_unbounded,
                                     following_window_in_days.value * mult_factor,
                                     following_window_in_days.is_unbounded,
                                     min_periods,
                                     aggr,
                                     mr);
  } else {
    return group_offsets.empty()
             ? time_range_window_DESC(input,
                                      timestamp_column,
                                      preceding_window_in_days.value * mult_factor,
                                      preceding_window_in_days.is_unbounded,
                                      following_window_in_days.value * mult_factor,
                                      following_window_in_days.is_unbounded,
                                      min_periods,
                                      aggr,
                                      mr)
             : time_range_window_DESC(input,
                                      timestamp_column,
                                      group_offsets,
                                      group_labels,
                                      preceding_window_in_days.value * mult_factor,
                                      preceding_window_in_days.is_unbounded,
                                      following_window_in_days.value * mult_factor,
                                      following_window_in_days.is_unbounded,
                                      min_periods,
                                      aggr,
                                      mr);
  }
}

}  // namespace

std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          size_type preceding_window_in_days,
                                                          size_type following_window_in_days,
                                                          size_type min_periods,
                                                          std::unique_ptr<aggregation> const& aggr,
                                                          rmm::mr::device_memory_resource* mr)
{
  return grouped_time_range_rolling_window(group_keys,
                                           timestamp_column,
                                           timestamp_order,
                                           input,
                                           window_bounds::get(preceding_window_in_days),
                                           window_bounds::get(following_window_in_days),
                                           min_periods,
                                           aggr,
                                           mr);
}

std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          window_bounds preceding_window_in_days,
                                                          window_bounds following_window_in_days,
                                                          size_type min_periods,
                                                          std::unique_ptr<aggregation> const& aggr,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) return empty_like(input);

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;
  using index_vector        = sort_groupby_helper::index_vector;

  index_vector group_offsets, group_labels;
  if (group_keys.num_columns() > 0) {
    sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
    group_offsets = helper.group_offsets();
    group_labels  = helper.group_labels();
  }

  // Assumes that `timestamp_column` is actually of a timestamp type.
  CUDF_EXPECTS(is_supported_range_frame_unit(timestamp_column.type()),
               "Unsupported data-type for `timestamp`-based rolling window operation!");

  return timestamp_column.type().id() == cudf::type_id::TIMESTAMP_DAYS
           ? grouped_time_range_rolling_window_impl<int32_t>(input,
                                                             timestamp_column,
                                                             timestamp_order,
                                                             group_offsets,
                                                             group_labels,
                                                             preceding_window_in_days,
                                                             following_window_in_days,
                                                             min_periods,
                                                             aggr,
                                                             mr)
           : grouped_time_range_rolling_window_impl<int64_t>(input,
                                                             timestamp_column,
                                                             timestamp_order,
                                                             group_offsets,
                                                             group_labels,
                                                             preceding_window_in_days,
                                                             following_window_in_days,
                                                             min_periods,
                                                             aggr,
                                                             mr);
}

}  // namespace cudf
