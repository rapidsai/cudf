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
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/aggregation.hpp>
#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <jit/type.h>
#include <jit/launcher.h>
#include <jit/parser.h>
#include <rolling/jit/code/code.h>

#include <types.hpp.jit>
#include <bit.hpp.jit>

#include <rmm/device_scalar.hpp>

#include <memory>

namespace cudf {
namespace experimental {

namespace detail {

namespace { // anonymous

/**
 * @brief Computes the rolling window function
 *
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam is_mean Compute mean=sum/count across all valid elements in the window
 * @tparam block_size CUDA block size for the kernel
 * @tparam has_nulls true if the input column has nulls
 * @tparam WindowIterator iterator type (inferred)
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
          typename WindowIterator>
__launch_bounds__(block_size)
__global__
void gpu_rolling(column_device_view input,
                 mutable_column_device_view output,
                 size_type * __restrict__ output_valid_count,
                 WindowIterator preceding_window_begin,
                 WindowIterator following_window_begin,
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
    size_type start = max(0, i - preceding_window);
    size_type end = min(input.size(), i + following_window + 1);
    size_type start_index = min(start, end);
    size_type end_index = max(start, end);

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

  template <typename T, typename agg_op, aggregation::Kind op, typename WindowIterator>
  std::enable_if_t<cudf::detail::is_supported<T, agg_op,
                                  op, op == aggregation::MEAN>(), std::unique_ptr<column>>
  launch(column_view const& input,
         WindowIterator preceding_window_begin,
         WindowIterator following_window_begin,
         size_type min_periods,
         std::unique_ptr<aggregation> const& agg,
         rmm::mr::device_memory_resource *mr,
         cudaStream_t stream) {

    if (input.is_empty()) return empty_like(input);

    cudf::nvtx::range_push("CUDF_ROLLING_WINDOW", cudf::nvtx::color::ORANGE);

    min_periods = std::max(min_periods, 1);

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

  template <typename T, typename agg_op, aggregation::Kind op, typename WindowIterator>
  std::enable_if_t<!cudf::detail::is_supported<T, agg_op,
                                  op, op == aggregation::MEAN>(), std::unique_ptr<column>>
  launch (column_view const& input,
          WindowIterator preceding_window_begin,
          WindowIterator following_window_begin,
          size_type min_periods,
          std::unique_ptr<aggregation> const& agg,
          rmm::mr::device_memory_resource *mr,
          cudaStream_t stream) {
      CUDF_FAIL("Aggregation operator and/or input type combination is invalid");
  }


  template<aggregation::Kind op, typename WindowIterator>
  std::enable_if_t<!(op == aggregation::MEAN), std::unique_ptr<column>>
  operator()(column_view const& input,
                                     WindowIterator preceding_window_begin,
                                     WindowIterator following_window_begin,
                                     size_type min_periods,
                                     std::unique_ptr<aggregation> const& agg,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream)
  {
      return launch <InputType, typename corresponding_operator<op>::type, op, WindowIterator> (
              input,
              preceding_window_begin,
              following_window_begin,
              min_periods,
              agg,
              mr,
              stream);
  }

  template<aggregation::Kind op, typename WindowIterator>
  std::enable_if_t<(op == aggregation::MEAN), std::unique_ptr<column>>
  operator()(column_view const& input,
                                     WindowIterator preceding_window_begin,
                                     WindowIterator following_window_begin,
                                     size_type min_periods,
                                     std::unique_ptr<aggregation> const& agg,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {

      return launch <InputType, cudf::DeviceSum, op, WindowIterator> (
              input,
              preceding_window_begin,
              following_window_begin,
              min_periods,
              agg,
              mr,
              stream);
  }


};

struct dispatch_rolling {
    template <typename T, typename WindowIterator>
    std::unique_ptr<column> operator()(column_view const& input,
                                     WindowIterator preceding_window_begin,
                                     WindowIterator following_window_begin,
                                     size_type min_periods,
                                     std::unique_ptr<aggregation> const& agg,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {

        return aggregation_dispatcher(agg->kind, rolling_window_launcher<T>{},
                                      input,
                                      preceding_window_begin, following_window_begin,
                                      min_periods, agg, mr, stream);
    }
};

} // namespace anonymous

// Applies a user-defined rolling window function to the values in a column.
template <bool static_window, typename WindowIterator>
std::unique_ptr<column> rolling_window_udf(column_view const &input,
                                           WindowIterator preceding_window,
                                           WindowIterator following_window,
                                           size_type min_periods,
                                           std::unique_ptr<aggregation> const& agg,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.has_nulls())
    CUDF_FAIL("Currently the UDF version of rolling window does NOT support inputs with nulls.");

  cudf::nvtx::range_push("CUDF_ROLLING_WINDOW", cudf::nvtx::color::ORANGE);

  min_periods = std::max(min_periods, 1);

  auto udf_agg = static_cast<udf_aggregation*>(agg.get());

  std::string hash = "prog_experimental_rolling." 
    + std::to_string(std::hash<std::string>{}(udf_agg->_source));
  
  std::string cuda_source;
  switch(udf_agg->kind){
    case aggregation::Kind::PTX:
      cuda_source = cudf::experimental::rolling::jit::code::kernel_headers;
      cuda_source += cudf::jit::parse_single_function_ptx(udf_agg->_source, udf_agg->_function_name,
                                                          cudf::jit::get_type_name(udf_agg->_output_type),
                                                          {0, 5}); // args 0 and 5 are pointers.
      cuda_source += cudf::experimental::rolling::jit::code::kernel;
      break; 
    case aggregation::Kind::CUDA:
      cuda_source = cudf::experimental::rolling::jit::code::kernel_headers;
      cuda_source += cudf::jit::parse_single_function_cuda(udf_agg->_source, udf_agg->_function_name);
      cuda_source += cudf::experimental::rolling::jit::code::kernel;
      break;
    default:
      CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(udf_agg->_output_type, input.size(),
                                                       cudf::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  rmm::device_scalar<size_type> device_valid_count{0, stream};

  // Launch the jitify kernel
  cudf::jit::launcher(hash, cuda_source,
                      { cudf_types_hpp, cudf_utilities_bit_hpp,
                        cudf::experimental::rolling::jit::code::operation_h },
                      { "-std=c++14", "-w" }, nullptr, stream)
    .set_kernel_inst("gpu_rolling_new", // name of the kernel we are launching
                      { cudf::jit::get_type_name(input.type()), // list of template arguments
                        cudf::jit::get_type_name(output->type()),
                        udf_agg->_operator_name,
                        static_window ? "cudf::size_type" : "cudf::size_type*"})
    .launch(input.size(), cudf::jit::get_data_ptr(input), input.null_mask(),
            cudf::jit::get_data_ptr(output_view), output_view.null_mask(),
            device_valid_count.data(), preceding_window, following_window, min_periods);

  output->set_null_count(output->size() - device_valid_count.value(stream));

  // check the stream for debugging
  CHECK_CUDA(stream);

  cudf::nvtx::range_pop();

  return output;
}

// Applies a rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       WindowIterator preceding_window_begin,
                                       WindowIterator following_window_begin,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  return cudf::experimental::type_dispatcher(input.type(),
                                             dispatch_rolling{},
                                             input,
                                             preceding_window_begin,
                                             following_window_begin,
                                             min_periods, agg, mr, stream);

}

} // namespace detail

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  if (agg->kind == aggregation::CUDA || agg->kind == aggregation::PTX) {
    return cudf::experimental::detail::rolling_window_udf<true>(input,
                                                                preceding_window,
                                                                following_window,
                                                                min_periods, agg, mr, 0);
  } else {
    auto preceding_window_begin = thrust::make_constant_iterator(preceding_window);
    auto following_window_begin = thrust::make_constant_iterator(following_window);

    return cudf::experimental::detail::rolling_window(input,
                                                      preceding_window_begin,
                                                      following_window_begin,
                                                      min_periods, agg, mr, 0);
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
  if (preceding_window.size() == 0 || following_window.size() == 0) return empty_like(input);

  CUDF_EXPECTS(preceding_window.type().id() == INT32 && following_window.type().id() == INT32,
               "preceding_window/following_window must have INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  if (agg->kind == aggregation::CUDA || agg->kind == aggregation::PTX) {
    return cudf::experimental::detail::rolling_window_udf<false>(input,
                                                                 preceding_window.begin<size_type>(),
                                                                 following_window.begin<size_type>(),
                                                                 min_periods, agg, mr, 0);
  } else {
    return cudf::experimental::detail::rolling_window(input, 
                                                      preceding_window.begin<size_type>(),
                                                      following_window.begin<size_type>(),
                                                      min_periods, agg, mr, 0);
  }
}

} // namespace experimental 
} // namespace cudf
