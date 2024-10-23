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
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/hyperloglog_ref.cuh>
#include <thrust/sequence.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

// The number of bits that is required per register.
// This is from Spark `HyperLogLogPlusPlusHelper.scala`: val REGISTER_SIZE = 6
// it's precision in cuCollection
constexpr int num_bits_per_register = 6;

constexpr int registers_per_long = 64 / num_bits_per_register;

/**
 * This function groups HLL sketch without using thrust::reduce_by_key.
 * It leverage the atomic_ref to do the sync.
 */
template <typename T, typename Iterator>
CUDF_KERNEL void compute_hll(column_device_view const d_values,   // input values
                             Iterator values_iter,                // input values iterator
                             int const num_registers_per_sketch,  // num registers per sketch
                             size_type const* d_group_labels,     // group labels
                             int32_t* sketches_output             // output: integer array
)
{
  auto const row_idx         = cudf::detail::grid_1d::global_thread_id();
  auto const group_idx       = d_group_labels[row_idx];
  auto hll_sketch_output_ptr = sketches_output + group_idx * num_registers_per_sketch;

  // refer to the sketch, Note: multi-threads handling the same group refer to the same sketch
  cuco::hyperloglog_ref<T, cuda::thread_scope_device> hll(cuda::std::span<cuda::std::byte>(
    reinterpret_cast<cuda::std::byte*>(hll_sketch_output_ptr), num_registers_per_sketch * 4));

  // the `add` for multiple threads is synchronized in coCo, code is like:
  // cuda::atomic_ref.fetch_max
  if (!d_values.is_null(row_idx)) { hll.add(values_iter[row_idx]); }
}

struct hll_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, std::unique_ptr<column>> operator()(
    column_view const& values,
    size_type num_groups,
    cudf::device_span<size_type const> group_labels,
    int const num_registers_per_sketch,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // use integer array to save hll sketches; one hll sketch uses num_registers_per_sketch integers
    auto hll_col = make_numeric_column(data_type{type_id::INT32},
                                       num_groups * num_registers_per_sketch,
                                       mask_state::ALL_VALID,
                                       stream);

    auto const d_values          = *column_device_view::create(values, stream);
    auto const sketch_output_ptr = hll_col->mutable_view().data<int32_t>();

    // clear memory for sketches
    CUDF_CUDA_TRY(
      cudaMemsetAsync(sketch_output_ptr, 0, num_groups * num_registers_per_sketch * 4, stream));

    static constexpr int block_size = 256;
    auto const num_blocks           = cudf::util::div_rounding_up_safe(values.size(), block_size);
    if (!cudf::is_dictionary(values.type())) {
      auto const values_iter = d_values.begin<T>();
      compute_hll<T, decltype(values_iter)><<<num_blocks, block_size, 0, stream.value()>>>(
        d_values, values_iter, num_registers_per_sketch, group_labels.data(), sketch_output_ptr);
    } else {
      auto const values_iter = cudf::dictionary::detail::make_dictionary_iterator<T>(d_values);
      compute_hll<T, decltype(values_iter)><<<num_blocks, block_size, 0, stream.value()>>>(
        d_values, values_iter, num_registers_per_sketch, group_labels.data(), sketch_output_ptr);
    }

    return hll_col;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic_v<T>, std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Only numeric types are supported in hll groupby aggregation");
  }

  // TODO only handle numeric types, need to add string and nested types
};

struct compact_functor {
  // input
  int32_t const* input_registers;

  // one integer stores one register for input
  int num_registers_per_sketch;

  // struct of long columns
  column_device_view output;

  __device__ void operator()(size_type row_idx)
  {
    int num_longs_per_sketch =
      cudf::util::div_rounding_up_safe(num_registers_per_sketch, registers_per_long);

    cudf::detail::structs_column_device_view output_struct(output);

    for (auto i = 0; i < num_longs_per_sketch; i++) {
      long packed = 0;
      for (auto j = 0; j < registers_per_long; j++) {
        long reg = input_registers[row_idx * num_registers_per_sketch + i * registers_per_long + j];
        packed |= (reg << (j * num_bits_per_register));
      }

      auto long_ptr     = const_cast<int64_t*>(output_struct.get_sliced_child(i).data<int64_t>());
      long_ptr[row_idx] = packed;
    }
  }
};

/**
 * Compress HLL sketchs like Spark does
 * The input `uncompressed_sketches` is a int32 column with N of int32 representing a sketch
 * Spark result layout is multiple long columns with each long column compressing bits.
 *
 * e.g.:
 * register uses 4 bits, register indices are: 0, 1,
 * and number of zeros are 9, 9. The bits for value 9 are 1001.
 *
 * the input bits are:
 * |0000 0000 0000 0000 0000 0000 0000 1001 | 0000 0000 0000 0000 0000 0000 0000 1001 |
 * Each int saves 1 register, the 24 of 32 bits can be compacted
 *
 * the Spark bits are:
 * |0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 1001 1001 |
 * The end of the 8 bits save the bits, Each long can store up to 16 registers
 */
std::unique_ptr<column> compact(column_view const& uncompressed_sketches,
                                int const num_registers_per_sketch,
                                rmm::cuda_stream_view stream)
{
  // Consistent with Spark
  int num_longs_per_sketch = num_registers_per_sketch / registers_per_long + 1;

  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto i = 0; i < num_longs_per_sketch; i++) {
    auto long_ptr = make_numeric_column(
      data_type{type_id::INT64}, uncompressed_sketches.size(), mask_state::ALL_VALID, stream);
    children.push_back(std::move(long_ptr));
  }

  auto result          = cudf::make_structs_column(uncompressed_sketches.size(),
                                          std::move(children),
                                          0,                     // null count
                                          rmm::device_buffer{},  // null mask
                                          stream);
  auto uncompressed_dv = cudf::column_device_view::create(uncompressed_sketches);
  auto result_dv       = cudf::column_device_view::create(result->view());

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    uncompressed_sketches.size(),
    compact_functor{uncompressed_dv->data<int32_t>(), num_registers_per_sketch, *result_dv});

  return result;
}

}  // namespace

/**
 * Compute hyper log log against the input values and merge the sketches in the same group.
 * Output is a struct column with multiple long columns which are consistent with Spark.
 */
std::unique_ptr<column> group_hyper_log_log(column_view const& values,
                                            size_type num_groups,
                                            cudf::device_span<size_type const> group_labels,
                                            int const num_registers_per_sketch,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();

  auto uncompressed_sketches = type_dispatcher(values_type,
                                               hll_functor{},
                                               values,
                                               num_groups,
                                               group_labels,
                                               num_registers_per_sketch,
                                               stream,
                                               mr);

  return compact(uncompressed_sketches->view(), num_registers_per_sketch, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
