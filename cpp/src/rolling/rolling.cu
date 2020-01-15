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

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/aggregation.hpp>
#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <rmm/device_scalar.hpp>

#include <memory>

namespace cudf {
namespace experimental {

namespace detail {

namespace { // anonymous

/**
 * @brief Computes the rolling window function for non sting_view
 *
 * @tparam InputType  Datatype of values pointed to by the pointers
 * @tparam OutputType Datatype of values expected at the end
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam op The aggreagtion expected to be carried out
 * @tparam block_size CUDA block size for the kernel
 * @tparam is_count_op true if op == aggregation::COUNT, else false
 * @tparam source_has_nulls true if the input column has nulls
 * @tparam WindowIterator iterator type (inferred)
 * @param input Input column device view
 * @param output Output column device view
 *
 * @param preceding_window_begin[in] Rolling window size iterator, accumulates from
 *                in_col[i-preceding_window] to in_col[i] inclusive
 * @param following_window_begin[in] Rolling window size iterator in the forward
 *                direction, accumulates from in_col[i] to
 *                in_col[i+following_window] inclusive
 * @param min_periods[in]  Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask
 * @param identity Identity value with InputType
 */
 template <typename InputType, typename OutputType, typename agg_op, aggregation::Kind op, int block_size, bool is_count_op, bool source_has_nulls,
           typename WindowIterator, std::enable_if_t<!std::is_same<InputType, cudf::string_view>::value>* = nullptr>
 __launch_bounds__(block_size)
 __global__
 void gpu_rolling(column_device_view input,
         mutable_column_device_view output,
         size_type * __restrict__ output_valid_count,
         WindowIterator preceding_window_begin,
         WindowIterator following_window_begin,
         size_type min_periods,
         InputType identity)
 {
     size_type i = blockIdx.x * block_size + threadIdx.x;
     size_type stride = block_size * gridDim.x;

     size_type warp_valid_count{0};

     auto active_threads = __ballot_sync(0xffffffff, i < input.size());
     while(i < input.size())
     {
         OutputType val = agg_op::template identity<OutputType>();
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
             if (!source_has_nulls || input.is_valid(j)) {
                 // Element type and output type are different for COUNT
                 OutputType element = (op == aggregation::COUNT) ? InputType{0} : input.element<InputType>(j);
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
             cudf::detail::store_output_functor<OutputType, op == aggregation::MEAN>{}(output.element<OutputType>(i),
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

/**
 * @brief Computes the rolling window function for string_view
 *
 * For Min and Max, this generates gather_map rather than actual
 * output
 *
 * @tparam InputType  Datatype of values pointed to by the pointers
 * @tparam OutputType Datatype of values expected at the end
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam op The aggreagtion expected to be carried out
 * @tparam block_size CUDA block size for the kernel
 * @tparam is_count_op true if op == aggregation::COUNT, else false
 * @tparam source_has_nulls true if the input column has nulls
 * @tparam WindowIterator iterator type (inferred)
 * @param input Input column device view
 * @param output Output column device view
 *
 * @param preceding_window_begin[in] Rolling window size iterator, accumulates from
 *                in_col[i-preceding_window] to in_col[i] inclusive
 * @param following_window_begin[in] Rolling window size iterator in the forward
 *                direction, accumulates from in_col[i] to
 *                in_col[i+following_window] inclusive
 * @param min_periods[in]  Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask
 * @param identity Identity value with InputType
 */
template <typename InputType, typename OutputType, typename agg_op, aggregation::Kind op, int block_size, bool is_count_op, bool source_has_nulls,
          typename WindowIterator, std::enable_if_t<std::is_same<InputType, cudf::string_view>::value>* = nullptr>
__launch_bounds__(block_size)
__global__
void gpu_rolling(column_device_view input,
        mutable_column_device_view output,
        size_type * __restrict__ output_valid_count,
        WindowIterator preceding_window_begin,
        WindowIterator following_window_begin,
        size_type min_periods,
        InputType identity)
{
    size_type i = blockIdx.x * block_size + threadIdx.x;
    size_type stride = block_size * gridDim.x;

    size_type warp_valid_count{0};

    auto active_threads = __ballot_sync(0xffffffff, i < input.size());
    while(i < input.size())
    {
        InputType val = identity;
        OutputType val_index = (op == aggregation::ARGMIN)? ARGMIN_SENTINEL : ARGMAX_SENTINEL;
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
            if (!source_has_nulls || input.is_valid(j)) {
                if (is_count_op) {
                    count++;
                } else {
                    // Element type and output type are different for COUNT
                    InputType element = input.element<InputType>(j);
                    val = agg_op{}(element, val);
                    if (val == element) {
                        val_index = j;
                    }
                    count++;
                }
            }
        }

        // check if we have enough input samples
        bool output_is_valid = (count >= min_periods);

        // set the mask
        // We can't have gather map being created for Min and Max for string_view to be null
        cudf::bitmask_type result_mask{__ballot_sync(active_threads, (is_count_op)? output_is_valid: true)};

        // only one thread writes the mask
        if (0 == threadIdx.x % cudf::experimental::detail::warp_size) {
            output.set_mask_word(cudf::word_index(i), result_mask);
            warp_valid_count += __popc(result_mask);
        }

        // store the output value, one per thread
        if (output_is_valid) {
            output.element<OutputType>(i) = (is_count_op)? count: val_index;
        } else {
            // This will help identify null elements while gathering for Min and Max
            // In case of count, this would be null, so doesn't matter.
            output.element<OutputType>(i) = -1;
        }

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

struct rolling_window_launcher
{
    template<typename InputType, typename agg_op, aggregation::Kind op, typename WindowIterator, bool is_count_op=false>
    void launch(column_view const& input,
                 mutable_column_view& output,
                 WindowIterator preceding_window_begin,
                 WindowIterator following_window_begin,
                 size_type min_periods,
                 cudaStream_t stream) 
        {
        constexpr cudf::size_type block_size = 256;
        cudf::experimental::detail::grid_1d grid(input.size(), block_size);

        auto input_device_view = column_device_view::create(input, stream);
        auto output_device_view = mutable_column_device_view::create(output, stream);

        rmm::device_scalar<size_type> device_valid_count{0, stream};


        if (input.has_nulls()) {
                gpu_rolling<InputType, target_type_t<InputType, op>, agg_op, op, block_size, is_count_op, true><<<grid.num_blocks, block_size, 0, stream>>>
                    (*input_device_view, *output_device_view, device_valid_count.data(),
                     preceding_window_begin, following_window_begin, min_periods, agg_op::template identity<InputType>());
        } else {
                gpu_rolling<InputType, target_type_t<InputType, op>, agg_op, op, block_size, is_count_op, false><<<grid.num_blocks, block_size, 0, stream>>>
                    (*input_device_view, *output_device_view, device_valid_count.data(),
                     preceding_window_begin, following_window_begin, min_periods, agg_op::template identity<InputType>());
        }

        output.set_null_count(output.size() - device_valid_count.value(stream));
    }

  template<typename T, typename agg_op, aggregation::Kind op, typename WindowIterator,
    std::enable_if_t<cudf::detail::is_supported<T, agg_op,
                                                 op == aggregation::MEAN>()>* = nullptr>
  std::unique_ptr<column> dispatch_aggregation_type(column_view const& input,
                                                    WindowIterator preceding_window_begin,
                                                    WindowIterator following_window_begin,
                                                    size_type min_periods,
                                                    rmm::mr::device_memory_resource *mr,
                                                    cudaStream_t stream)
  {
    if (input.is_empty()) return empty_like(input);

    cudf::nvtx::range_push("CUDF_ROLLING_WINDOW", cudf::nvtx::color::ORANGE);

    auto output = make_fixed_width_column(target_type(input.type(), op), input.size(),
          UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();
    launch<T, agg_op, op, WindowIterator>(input, output_view, preceding_window_begin,
                                                following_window_begin, min_periods, stream);

    // check the stream for debugging
    CHECK_CUDA(stream);

    cudf::nvtx::range_pop();

    return output;
  }

  template<typename T, typename agg_op, aggregation::Kind op, typename WindowIterator,
    std::enable_if_t<cudf::detail::is_string_supported<T, op>()>* = nullptr>
  std::unique_ptr<column> dispatch_aggregation_type(column_view const& input,
                                                    WindowIterator preceding_window_begin,
                                                    WindowIterator following_window_begin,
                                                    size_type min_periods,
                                                    rmm::mr::device_memory_resource *mr,
                                                    cudaStream_t stream)
  {
    auto output = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<size_type>()},
            input.size(), cudf::UNINITIALIZED, stream, mr);

    cudf::mutable_column_view output_view = output->mutable_view();
    if(op == aggregation::MIN) {
        launch<T, agg_op, aggregation::ARGMIN, WindowIterator>(input, output_view, preceding_window_begin,
                                                    following_window_begin, min_periods, stream);
    } else if(op == aggregation::MAX) {
        launch<T, agg_op, aggregation::ARGMAX, WindowIterator>(input, output_view, preceding_window_begin,
                                                    following_window_begin, min_periods, stream);
    } else {
        launch<T, agg_op, aggregation::COUNT, WindowIterator, true>(input, output_view, preceding_window_begin,
                                                    following_window_begin, min_periods, stream);
    }

    // check the stream for debugging
    CHECK_CUDA(stream);

    cudf::nvtx::range_pop();

    // If aggregation operation is MIN or MAX, then the output we got is a scatter map
    if((op == aggregation::MIN) or (op == aggregation::MAX)) {
        // The rows that represent null elements will be having negative values in gather map,
        // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
        auto output_table = detail::gather(table_view{{input}}, output->view(), false, true, false, mr, stream);
        return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));;
    }
    
    return output;
  }

  /**
   * @brief If we cannot perform aggregation on this type then throw an error
   */
  template<typename T, typename agg_op, aggregation::Kind op, typename WindowIterator,
    std::enable_if_t<!cudf::detail::is_supported<T, agg_op,
                                                 op == aggregation::MEAN>() and !cudf::detail::is_string_supported<T, op>()>* = nullptr>
  std::unique_ptr<column> dispatch_aggregation_type(column_view const& input,
                                                    WindowIterator preceding_window_begin,
                                                    WindowIterator following_window_begin,
                                                    size_type min_periods,
                                                    rmm::mr::device_memory_resource *mr,
                                                    cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported column type/operation combo. Only `min` and `max` are supported for "
              "non-arithmetic types for aggregations.");
  }

  /**
   * @brief Helper function for gdf_rolling. Deduces the type of the
   * aggregation column and type and calls another function to invoke the
   * rolling window kernel.
   */
  template <typename T, typename WindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     WindowIterator preceding_window_begin,
                                     WindowIterator following_window_begin,
                                     size_type min_periods,
                                     std::unique_ptr<aggregation> const& aggr,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream)
  {
    switch (aggr->kind) {
    case aggregation::SUM:
      return dispatch_aggregation_type<T, corresponding_operator<aggregation::SUM>::type,
                                       aggregation::SUM>(input, preceding_window_begin,
                                                               following_window_begin, min_periods,
                                                               mr, stream);
    case aggregation::MIN:
      return dispatch_aggregation_type<T, corresponding_operator<aggregation::MIN>::type,
                                       aggregation::MIN>(input, preceding_window_begin,
                                                              following_window_begin, min_periods,
                                                              mr, stream);
    case aggregation::MAX:
      return dispatch_aggregation_type<T, corresponding_operator<aggregation::MAX>::type,
                                       aggregation::MAX>(input, preceding_window_begin,
                                                              following_window_begin, min_periods,
                                                              mr, stream);
    case aggregation::COUNT:
      // for count, use size_type rather than the input type (never load the input)
      return dispatch_aggregation_type<cudf::size_type, cudf::DeviceCount,
                                       aggregation::COUNT>(input, preceding_window_begin,
                                                                following_window_begin, min_periods,
                                                                mr, stream);
    case aggregation::MEAN:
      return dispatch_aggregation_type<T, corresponding_operator<aggregation::SUM>::type,
                                       aggregation::MEAN>(input, preceding_window_begin,
                                                               following_window_begin, min_periods,
                                                               mr, stream);
    default:
      // TODO: need a nice way to convert enums to strings, same would be useful for groupby
      CUDF_FAIL("Rolling aggregation function not implemented");
    }
  }
};

} // namespace anonymous

// Applies a rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       WindowIterator preceding_window_begin,
                                       WindowIterator following_window_begin,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  return cudf::experimental::type_dispatcher(input.type(),
                                             rolling_window_launcher{},
                                             input, preceding_window_begin, following_window_begin,
                                             min_periods, aggr, mr, stream);
}

} // namespace detail

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS((preceding_window >= 0) && (following_window >= 0) && (min_periods >= 0),
               "Window sizes and min periods must be non-negative");

  auto preceding_window_begin = thrust::make_constant_iterator(preceding_window);
  auto following_window_begin = thrust::make_constant_iterator(following_window);

  return cudf::experimental::detail::rolling_window(input, preceding_window_begin,
                                                    following_window_begin, min_periods, aggr, mr, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  if (preceding_window.size() == 0 || following_window.size() == 0) return empty_like(input);

  CUDF_EXPECTS(preceding_window.type().id() == INT32 && following_window.type().id() == INT32,
               "preceding_window/following_window must have INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, preceding_window.begin<size_type>(),
                                                    following_window.begin<size_type>(),
                                                    min_periods, aggr, mr, 0);
}

} // namespace experimental 
} // namespace cudf
