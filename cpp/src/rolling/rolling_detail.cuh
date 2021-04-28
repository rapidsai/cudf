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

#include "rolling/rolling_collect_list.cuh"
#include "rolling/rolling_detail.hpp"
#include "rolling/rolling_jit_detail.hpp"

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

namespace cudf {

namespace detail {

namespace {  // anonymous

template <typename InputType, aggregation::Kind op>
struct DeviceRolling {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = op>
  static constexpr bool is_supported()
  {
    return cudf::detail::is_valid_aggregation<T, O>() && has_corresponding_operator<O>() &&
             // MIN/MAX supports all fixed width types
             ((op == aggregation::MIN || op == aggregation::MAX) && cudf::is_fixed_width<T>()) ||

           // SUM supports all fixed width types except timestamps
           ((op == aggregation::SUM) && (cudf::is_fixed_width<T>() && !cudf::is_timestamp<T>())) ||

           // MEAN supports numeric and duration
           ((op == aggregation::MEAN) && (cudf::is_numeric<T>() || cudf::is_duration<T>()));
  }

  // operations we do support
  template <typename T = InputType, aggregation::Kind O = op>
  DeviceRolling(size_type _min_periods, typename std::enable_if_t<is_supported<T, O>()>* = nullptr)
    : min_periods(_min_periods)
  {
  }

  // operations we don't support
  template <typename T = InputType, aggregation::Kind O = op>
  DeviceRolling(size_type _min_periods, typename std::enable_if_t<!is_supported<T, O>()>* = nullptr)
    : min_periods(_min_periods)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  template <typename OutputType, bool has_nulls, typename T = InputType, aggregation::Kind O = op>
  std::enable_if_t<is_supported<T, O>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& ignored_default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    using AggOp = typename corresponding_operator<O>::type;
    AggOp agg_op;

    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;
    OutputType val                 = AggOp::template identity<OutputType>();

    for (size_type j = start_index; j < end_index; j++) {
      if (!has_nulls || input.is_valid(j)) {
        OutputType element = input.element<device_storage_type_t<InputType>>(j);
        val                = agg_op(element, val);
        count++;
      }
    }

    bool output_is_valid = (count >= min_periods);

    // store the output value, one per thread
    cudf::detail::rolling_store_output_functor<OutputType, op == aggregation::MEAN>{}(
      output.element<OutputType>(current_index), val, count);

    return output_is_valid;
  }

  template <typename OutputType, bool has_nulls, typename T = InputType, aggregation::Kind O = op>
  std::enable_if_t<!is_supported<T, O>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& ignored_default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    return false;
  }
};

template <typename InputType, aggregation::Kind op>
struct DeviceRollingArgMinMax {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = op>
  static constexpr bool is_supported()
  {
    // strictly speaking, I think it would be ok to make this work
    // for comparable types as well.  but right now the only use case is
    // for MIN/MAX on strings.
    return std::is_same<T, cudf::string_view>::value;
  }

  DeviceRollingArgMinMax(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls, typename T = InputType, aggregation::Kind O = op>
  std::enable_if_t<is_supported<T, O>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& ignored_default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    using AggOp = typename corresponding_operator<O>::type;
    AggOp agg_op;

    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;
    InputType val                  = AggOp::template identity<InputType>();
    OutputType val_index = (op == aggregation::ARGMIN) ? ARGMIN_SENTINEL : ARGMAX_SENTINEL;

    for (size_type j = start_index; j < end_index; j++) {
      if (!has_nulls || input.is_valid(j)) {
        InputType element = input.element<InputType>(j);
        val               = agg_op(element, val);
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

  template <typename OutputType, bool has_nulls, typename T = InputType, aggregation::Kind O = op>
  std::enable_if_t<!is_supported<T, O>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& ignored_default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    return false;
  }
};

template <typename InputType>
struct DeviceRollingCountValid {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::COUNT_VALID>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingCountValid(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const& ignored_default_outputs,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
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
};

template <typename InputType>
struct DeviceRollingCountAll {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::COUNT_ALL>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingCountAll(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const& ignored_default_outputs,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    cudf::size_type count = end_index - start_index;

    bool output_is_valid                      = count >= min_periods;
    output.element<OutputType>(current_index) = count;

    return output_is_valid;
  }
};

template <typename InputType>
struct DeviceRollingRowNumber {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::ROW_NUMBER>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingRowNumber(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const& ignored_default_outputs,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    bool output_is_valid                      = end_index - start_index >= min_periods;
    output.element<OutputType>(current_index) = current_index - start_index + 1;

    return output_is_valid;
  }
};

template <typename InputType>
struct DeviceRollingLead {
  size_type row_offset;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::LEAD>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>();
  }

  template <typename T = InputType, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  DeviceRollingLead(size_type _row_offset) : row_offset(_row_offset)
  {
  }

  template <typename T = InputType, typename std::enable_if_t<!is_supported<T>()>* = nullptr>
  DeviceRollingLead(size_type _row_offset) : row_offset(_row_offset)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  template <typename OutputType, bool has_nulls, typename T = InputType>
  std::enable_if_t<is_supported<T>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    // Offsets have already been normalized.

    // Check if row is invalid.
    if (row_offset > (end_index - current_index - 1)) {
      // Invalid row marked. Use default value, if available.
      if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

      output.element<OutputType>(current_index) =
        default_outputs.element<OutputType>(current_index);
      return true;
    }

    // Not an invalid row.
    auto index   = current_index + row_offset;
    auto is_null = input.is_null(index);
    if (!is_null) {
      output.element<OutputType>(current_index) =
        input.element<device_storage_type_t<InputType>>(index);
    }
    return !is_null;
  }

  template <typename OutputType, bool has_nulls, typename T = InputType>
  std::enable_if_t<!is_supported<T>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    return false;
  }
};

template <typename InputType>
struct DeviceRollingLag {
  size_type row_offset;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::LAG>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>();
  }

  template <typename T = InputType, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  DeviceRollingLag(size_type _row_offset) : row_offset(_row_offset)
  {
  }

  template <typename T = InputType, typename std::enable_if_t<!is_supported<T>()>* = nullptr>
  DeviceRollingLag(size_type _row_offset) : row_offset(_row_offset)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  template <typename OutputType, bool has_nulls, typename T = InputType>
  std::enable_if_t<is_supported<T>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    // Offsets have already been normalized.

    // Check if row is invalid.
    if (row_offset > (current_index - start_index)) {
      // Invalid row marked. Use default value, if available.
      if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

      output.element<OutputType>(current_index) =
        default_outputs.element<OutputType>(current_index);
      return true;
    }

    // Not an invalid row.
    auto index   = current_index - row_offset;
    auto is_null = input.is_null(index);
    if (!is_null) {
      output.element<OutputType>(current_index) =
        input.element<device_storage_type_t<InputType>>(index);
    }
    return !is_null;
  }

  template <typename OutputType, bool has_nulls, typename T = InputType>
  std::enable_if_t<!is_supported<T>(), bool> __device__
  operator()(column_device_view const& input,
             column_device_view const& default_outputs,
             mutable_column_device_view& output,
             size_type start_index,
             size_type end_index,
             size_type current_index)
  {
    return false;
  }
};

template <typename InputType, aggregation::Kind op>
struct create_rolling_operator {
  // everything else
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRolling<InputType, op>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::ARGMIN> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingArgMinMax<InputType, aggregation::ARGMIN>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::ARGMAX> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingArgMinMax<InputType, aggregation::ARGMAX>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::COUNT_VALID> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingCountValid<InputType>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::COUNT_ALL> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingCountAll<InputType>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::ROW_NUMBER> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingRowNumber<InputType>{min_periods};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::LEAD> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingLead<InputType>{
      dynamic_cast<cudf::detail::lead_lag_aggregation const&>(agg).row_offset};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::LAG> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingLag<InputType>{
      dynamic_cast<cudf::detail::lead_lag_aggregation const&>(agg).row_offset};
  }
};


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
          aggregation::Kind op,
          int block_size,
          bool has_nulls,
          typename DeviceRollingOperator,
          typename PrecedingWindowIterator,
          typename FollowingWindowIterator>
__launch_bounds__(block_size) __global__
  void gpu_rolling(column_device_view input,
                   column_device_view default_outputs,
                   mutable_column_device_view output,
                   size_type* __restrict__ output_valid_count,
                   DeviceRollingOperator device_operator,
                   PrecedingWindowIterator preceding_window_begin,
                   FollowingWindowIterator following_window_begin)
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
    output_is_valid               = device_operator.template operator()<OutputType, has_nulls>(
      input, default_outputs, output, start_index, end_index, i);

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
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator,
            typename DeviceRollingOperator>
  std::unique_ptr<column> launch(cudf::data_type output_type,
                                 column_view const& input,
                                 column_view const& default_outputs,
                                 PrecedingWindowIterator preceding_window_begin,
                                 FollowingWindowIterator following_window_begin,
                                 DeviceRollingOperator device_operator,
                                 rolling_aggregation const& agg,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
  {
    auto output =
      make_fixed_width_column(output_type, input.size(), mask_state::UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();

    size_type valid_count{0};
    {
      using Type    = device_storage_type_t<InputType>;
      using OutType = device_storage_type_t<target_type_t<InputType, op>>;

      constexpr cudf::size_type block_size = 256;
      cudf::detail::grid_1d grid(input.size(), block_size);

      auto input_device_view           = column_device_view::create(input, stream);
      auto output_device_view          = mutable_column_device_view::create(output_view, stream);
      auto default_outputs_device_view = column_device_view::create(default_outputs, stream);

      rmm::device_scalar<size_type> device_valid_count{0, stream};

      if (input.has_nulls()) {
        gpu_rolling<Type, OutType, op, block_size, true>
          <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                               *default_outputs_device_view,
                                                               *output_device_view,
                                                               device_valid_count.data(),
                                                               device_operator,
                                                               preceding_window_begin,
                                                               following_window_begin);
      } else {
        gpu_rolling<Type, OutType, op, block_size, false>
          <<<grid.num_blocks, block_size, 0, stream.value()>>>(*input_device_view,
                                                               *default_outputs_device_view,
                                                               *output_device_view,
                                                               device_valid_count.data(),
                                                               device_operator,
                                                               preceding_window_begin,
                                                               following_window_begin);
      }

      valid_count = device_valid_count.value(stream);

      // check the stream for debugging
      CHECK_CUDA(stream.value());
    }

    output->set_null_count(output->size() - valid_count);

    return output;
  }

  // family 1 : operations that can be expressed with a generic rolling_window kernel
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!is_rolling_string_specialization<InputType, op>() &&
                     (op != aggregation::COLLECT_LIST),
                   std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& default_outputs,
             PrecedingWindowIterator preceding_window_begin,
             FollowingWindowIterator following_window_begin,
             int min_periods,
             rolling_aggregation const& agg,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    auto device_operator = create_rolling_operator<InputType, op>{}(min_periods, agg);
    return launch<op, PrecedingWindowIterator, FollowingWindowIterator, decltype(device_operator)>(
      target_type(input.type(), op),
      input,
      default_outputs,
      preceding_window_begin,
      following_window_begin,
      device_operator,
      agg,
      stream,
      mr);
  }

  // family 2 : operations that require the kernel to output indices instead of values, for
  // subsequent gathering
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<is_rolling_string_specialization<InputType, op>() &&
                     (op != aggregation::COLLECT_LIST),
                   std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& default_outputs,
             PrecedingWindowIterator preceding_window_begin,
             FollowingWindowIterator following_window_begin,
             int min_periods,
             rolling_aggregation const& agg,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    // since we are dealing with strings, we will change the aggregations to ones that
    // return indices of the MIN or MAX values instead of the values themselves, then do a
    // gather on the results as a postprocess.
    if (op == aggregation::MIN) {
      auto device_operator =
        create_rolling_operator<InputType, aggregation::ARGMIN>{}(min_periods, agg);
      auto gather_map = launch<aggregation::ARGMIN,
                               PrecedingWindowIterator,
                               FollowingWindowIterator,
                               decltype(device_operator)>(cudf::data_type{type_id::INT32},
                                                          input,
                                                          default_outputs,
                                                          preceding_window_begin,
                                                          following_window_begin,
                                                          device_operator,
                                                          agg,
                                                          stream,
                                                          mr);

      // The rows that represent null elements will be having negative values in gather map,
      // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
      auto output_table = detail::gather(table_view{{input}},
                                         gather_map->view(),
                                         cudf::out_of_bounds_policy::NULLIFY,
                                         detail::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
    } else if (op == aggregation::MAX) {
      auto device_operator =
        create_rolling_operator<InputType, aggregation::ARGMAX>{}(min_periods, agg);
      auto gather_map = launch<aggregation::ARGMAX,
                               PrecedingWindowIterator,
                               FollowingWindowIterator,
                               decltype(device_operator)>(cudf::data_type{type_id::INT32},
                                                          input,
                                                          default_outputs,
                                                          preceding_window_begin,
                                                          following_window_begin,
                                                          device_operator,
                                                          agg,
                                                          stream,
                                                          mr);

      // The rows that represent null elements will be having negative values in gather map,
      // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
      auto output_table = detail::gather(table_view{{input}},
                                         gather_map->view(),
                                         cudf::out_of_bounds_policy::NULLIFY,
                                         detail::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
    }
    CUDF_FAIL("MIN and MAX are the only supported aggregation types for string columns");
  }

  // family
  template <aggregation::Kind op, typename PrecedingIter, typename FollowingIter>
  std::enable_if_t<op == aggregation::COLLECT_LIST, std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& default_outputs,
    PrecedingIter preceding_begin_raw,
    FollowingIter following_begin_raw,
    size_type min_periods,
    rolling_aggregation const& agg,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    return rolling_collect_list(input,
                                default_outputs,
                                preceding_begin_raw,
                                following_begin_raw,
                                min_periods,
                                agg,
                                stream,
                                mr);
  }
};

struct dispatch_rolling {
  template <typename InputType, typename PrecedingWindowIterator, typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     PrecedingWindowIterator preceding_window_begin,
                                     FollowingWindowIterator following_window_begin,
                                     size_type min_periods,
                                     rolling_aggregation const& agg,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return aggregation_dispatcher(agg.kind,
                                  rolling_window_launcher<InputType>{},
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
                                           rolling_aggregation const& agg,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.has_nulls()) {
    CUDF_FAIL("Currently the UDF version of rolling window does NOT support inputs with nulls.");
  }

  min_periods = std::max(min_periods, 0);

  auto udf_agg = dynamic_cast<udf_aggregation const&>(agg);

  std::string hash = "prog_rolling." + std::to_string(std::hash<std::string>{}(udf_agg._source));

  std::string cuda_source;
  switch (udf_agg.kind) {
    case aggregation::Kind::PTX:
      cuda_source +=
        cudf::jit::parse_single_function_ptx(udf_agg._source,
                                             udf_agg._function_name,
                                             cudf::jit::get_type_name(udf_agg._output_type),
                                             {0, 5});  // args 0 and 5 are pointers.
      break;
    case aggregation::Kind::CUDA:
      cuda_source +=
        cudf::jit::parse_single_function_cuda(udf_agg._source, udf_agg._function_name);
      break;
    default: CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(
    udf_agg._output_type, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  rmm::device_scalar<size_type> device_valid_count{0, stream};

  std::string kernel_name =
    jitify2::reflection::Template("cudf::rolling::jit::gpu_rolling_new")  //
      .instantiate(cudf::jit::get_type_name(input.type()),  // list of template arguments
                   cudf::jit::get_type_name(output->type()),
                   udf_agg._operator_name,
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
 *                               aggregation const& agg,
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
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.is_empty()) { return empty_like(input); }

  if (cudf::is_dictionary(input.type())) {
    CUDF_EXPECTS(agg.kind == aggregation::COUNT_ALL || agg.kind == aggregation::COUNT_VALID ||
                   agg.kind == aggregation::ROW_NUMBER || agg.kind == aggregation::MIN ||
                   agg.kind == aggregation::MAX || agg.kind == aggregation::LEAD ||
                   agg.kind == aggregation::LAG,
                 "Invalid aggregation for dictionary column");
  }

  if (agg.kind != aggregation::LEAD && agg.kind != aggregation::LAG) {
    CUDF_EXPECTS(default_outputs.is_empty(),
                 "Only LEAD/LAG window functions support default values.");
  }

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
  if (agg.kind == aggregation::COUNT_ALL || agg.kind == aggregation::COUNT_VALID ||
      agg.kind == aggregation::ROW_NUMBER) {
    return output;
  }

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
