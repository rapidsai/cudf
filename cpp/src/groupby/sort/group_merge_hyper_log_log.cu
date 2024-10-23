/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

// // The number of bits that is required per register.
// // This is from Spark `HyperLogLogPlusPlusHelper`: val REGISTER_SIZE = 6
// // it's precision in cuCollection
// constexpr int num_bits_per_register = 6;

// constexpr int registers_per_long = 64 / num_bits_per_register;

// /**
//  * sketch memory: [int, int, int ... int]
//  * find the max value for each int for all sketch memories.
//  */
// struct merge_hll_fn {
//   int32_t const* registers_input;  // input
//   size_type const* d_group_offset;  // hlls offset for a specific group
//   int32_t* d_hll_bytes_output;       // output

//   __device__ void operator()(size_type group_idx)
//   {
//     // create a output hll to save result
//     auto hll_sketch_output_ptr = d_hll_bytes_output + group_idx * SKETCH_BYTES;
//     memset(hll_sketch_output_ptr, 0, SKETCH_BYTES);
//     int* sketch_ptr = reinterpret_cast<int*>(hll_sketch_output_ptr);

//     // merge all hll in the same group
//     auto hll_start_idx = d_group_offset[group_idx];
//     auto num_hlls      = d_group_offset[group_idx + 1] - hll_start_idx;

//     for (auto register_idx = 0; register_idx < SKETCH_INTS / 4; register_idx++) {
//       int curr_max = 0;
//       for (auto hll_idx = 0; hll_idx < num_hlls; hll_idx++) {
//         auto tmp_sketch_ptr = reinterpret_cast<int const*>(
//           registers_input + (hll_start_idx + hll_idx) * SKETCH_BYTES);
//         curr_max =
//           tmp_sketch_ptr[register_idx] > curr_max ? tmp_sketch_ptr[register_idx] : curr_max;
//       }
//       sketch_ptr[register_idx] = curr_max;
//     }
//   }
// };


// struct merge_functor {
//   // struct of long columns
//   column_device_view input;

//   // struct of long columns
//   column_device_view output;

//   // one integer stores one register for input
//   int num_registers_per_sketch;

//   __device__ void operator()(size_type row_idx)
//   {
//     int num_longs_per_sketch =
//       cudf::util::div_rounding_up_safe(num_registers_per_sketch, registers_per_long);

//     cudf::detail::structs_column_device_view output_struct(output);

//     for (auto i = 0; i < num_longs_per_sketch; i++) {
//       long packed = 0;
//       for (auto j = 0; j < registers_per_long; j++) {
//         long reg = input_registers[row_idx * num_registers_per_sketch + i * registers_per_long + j];
//         packed |= (reg << (j * num_bits_per_register));
//       }

//       auto long_ptr     = const_cast<int64_t*>(output_struct.get_sliced_child(i).data<int64_t>());
//       long_ptr[row_idx] = packed;
//     }
//   }
// };

// void merge_hll(size_type num_hlls,
//                          int32_t const* sketch,  // input
//                          int const num_registers_per_sketch,
//                          size_type const* d_grouop_offsets,  // group offsets
//                          size_type num_groups,               // num of groups
//                          int8_t* d_hll_bytes_output,         // output
//                          rmm::cuda_stream_view stream)
// {
//   constexpr int NUM_GROUPS_THRESHOLD = 32;

//   if (num_groups > NUM_GROUPS_THRESHOLD) {
//     thrust::for_each_n(rmm::exec_policy(stream),
//                        thrust::make_counting_iterator(0),
//                        num_groups,
//                        merge_hll_fn{d_hll_bytes_input, d_grouop_offsets, d_hll_bytes_output});
//   } else {
//     // TODO leverage shared mem
//   }
// }

std::unique_ptr<column> merge_hyper_log_log(column_view const& hll_input,  // struct<long, ..., long> column
                                            size_type num_groups,
                                            cudf::device_span<size_type const> group_lables,
                                            int const num_registers_per_sketch,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // make result struct column
  int num_long_columns = hll_input.num_children();
  std::vector<std::unique_ptr<cudf::column>> output_children;
  for (auto i = 0; i < num_long_columns; i++) {
    auto long_ptr = make_numeric_column(
      data_type{type_id::INT64}, num_groups, mask_state::ALL_VALID, stream);
    output_children.push_back(std::move(long_ptr));
  }
  auto result          = cudf::make_structs_column(num_groups,
                                          std::move(output_children),
                                          0,                     // null count
                                          rmm::device_buffer{},  // null mask
                                          stream);
  auto result_dv       = cudf::column_device_view::create(result->view());

  // merge hll
  // TODO
  // 

  return result;
}

}  // namespace

std::unique_ptr<column> group_merge_hyper_log_log(column_view const& values,
                                                  size_type num_groups,
                                                  cudf::device_span<size_type const> group_lables,
                                                  int const num_registers_per_sketch,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  // TODO assert values column is a struct<long, ..., long> column
  return merge_hyper_log_log(
    values, num_groups, group_lables, num_registers_per_sketch, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
