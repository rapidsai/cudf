/*
 * Copyright 2018 BlazingDB, Inc.

 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <cub/cub.cuh>

namespace {  // anonymous

static constexpr int BLOCK_SIZE = 256;

// return the new_value for output column at index `idx`
template <class T, bool replacement_has_nulls>
__device__ auto get_new_value(cudf::size_type idx,
                              const T* __restrict__ input_data,
                              const T* __restrict__ values_to_replace_begin,
                              const T* __restrict__ values_to_replace_end,
                              const T* __restrict__ d_replacement_values,
                              cudf::bitmask_type const* __restrict__ replacement_valid)
{
  auto found_ptr =
    thrust::find(thrust::seq, values_to_replace_begin, values_to_replace_end, input_data[idx]);
  T new_value{0};
  bool output_is_valid{true};

  if (found_ptr != values_to_replace_end) {
    auto d    = thrust::distance(values_to_replace_begin, found_ptr);
    new_value = d_replacement_values[d];
    if (replacement_has_nulls) { output_is_valid = cudf::bit_is_set(replacement_valid, d); }
  } else {
    new_value = input_data[idx];
  }
  return thrust::make_pair(new_value, output_is_valid);
}

__device__ int get_new_string_value(cudf::size_type idx,
                                    cudf::column_device_view& input,
                                    cudf::column_device_view& values_to_replace,
                                    cudf::column_device_view& replacement_values)
{
  cudf::string_view input_string = input.element<cudf::string_view>(idx);
  int match                      = -1;
  for (int i = 0; i < values_to_replace.size(); i++) {
    cudf::string_view value_string = values_to_replace.element<cudf::string_view>(i);
    if (input_string == value_string) {
      match = i;
      break;
    }
  }
  return match;
}

/*
 * Kernel which does the first pass of strings replace. It computes the output null_mask,
 * null_count, and the offsets.
 *
 * @param input The input column to replace strings in.
 * @param values_to_replace The string values to replace.
 * @param replacement The replacement values.
 * @param offsets The column which will contain the offsets of the new string column
 * @param indices Temporary column used to store the replacement indices
 * @param output_valid The output null_mask
 * @param output_valid_count The output valid count
 */
template <bool input_has_nulls, bool replacement_has_nulls>
__global__ void replace_strings_first_pass(cudf::column_device_view input,
                                           cudf::column_device_view values_to_replace,
                                           cudf::column_device_view replacement,
                                           cudf::mutable_column_device_view offsets,
                                           cudf::mutable_column_device_view indices,
                                           cudf::bitmask_type* output_valid,
                                           cudf::size_type* __restrict__ output_valid_count)
{
  cudf::size_type nrows = input.size();
  cudf::size_type i     = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t active_mask  = 0xffffffff;
  active_mask           = __ballot_sync(active_mask, i < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (i < nrows) {
    bool input_is_valid = true;

    if (input_has_nulls) input_is_valid = input.is_valid_nocheck(i);
    bool output_is_valid = input_is_valid;

    if (input_is_valid) {
      int result               = get_new_string_value(i, input, values_to_replace, replacement);
      cudf::string_view output = (result == -1) ? input.element<cudf::string_view>(i)
                                                : replacement.element<cudf::string_view>(result);
      offsets.data<cudf::size_type>()[i] = output.size_bytes();
      indices.data<cudf::size_type>()[i] = result;
      if (replacement_has_nulls && result != -1) {
        output_is_valid = replacement.is_valid_nocheck(result);
      }
    } else {
      offsets.data<cudf::size_type>()[i] = 0;
      indices.data<cudf::size_type>()[i] = -1;
    }

    uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
    if (0 == lane_id) {
      output_valid[cudf::word_index(i)] = bitmask;
      valid_sum += __popc(bitmask);
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }

  // Compute total valid count for this block and add it to global count
  uint32_t block_valid_count = cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
  // one thread computes and adds to output_valid_count
  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

/*
 * Kernel which does the second pass of strings replace. It copies the string data needed from input
 * and replacement into the new strings column chars column.
 * @param input The input column
 * @param replacement The replacement values
 * @param offsets The offsets column of the new strings column
 * @param strings The chars column of the new strings column
 * @param indices Temporary column used to store the replacement indices.
 */
template <bool input_has_nulls, bool replacement_has_nulls>
__global__ void replace_strings_second_pass(cudf::column_device_view input,
                                            cudf::column_device_view replacement,
                                            cudf::mutable_column_device_view offsets,
                                            cudf::mutable_column_device_view strings,
                                            cudf::mutable_column_device_view indices)
{
  cudf::size_type nrows = input.size();
  cudf::size_type i     = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < nrows) {
    bool output_is_valid = true;
    bool input_is_valid  = true;
    cudf::size_type idx  = indices.element<cudf::size_type>(i);

    if (input_has_nulls) {
      input_is_valid  = input.is_valid_nocheck(i);
      output_is_valid = input_is_valid;
    }
    if (replacement_has_nulls && idx != -1) { output_is_valid = replacement.is_valid_nocheck(idx); }
    if (output_is_valid) {
      cudf::string_view output = (idx == -1) ? input.element<cudf::string_view>(i)
                                             : replacement.element<cudf::string_view>(idx);
      std::memcpy(strings.data<char>() + offsets.data<cudf::size_type>()[i],
                  output.data(),
                  output.size_bytes());
    }

    i += blockDim.x * gridDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief Kernel that replaces elements from `output_data` given the following
 *        rule: replace all `values_to_replace[i]` in [values_to_replace_begin`,
 *        `values_to_replace_end`) present in `output_data` with `d_replacement_values[i]`.
 *
 * @tparam input_has_nulls `true` if output column has valid mask, `false` otherwise
 * @tparam replacement_has_nulls `true` if replacement_values column has valid mask, `false`
 * otherwise The input_has_nulls and replacement_has_nulls template parameters allows us to
 * specialize this kernel for the different scenario for performance without writing different
 * kernel.
 *
 * @param[in] input_data Device array with the data to be modified
 * @param[in] input_valid Valid mask associated with input_data
 * @param[out] output_data Device array to store the data from input_data
 * @param[out] output_valid Valid mask associated with output_data
 * @param[out] output_valid_count #valid in output column
 * @param[in] nrows # rows in `output_data`
 * @param[in] values_to_replace_begin Device pointer to the beginning of the sequence
 * of old values to be replaced
 * @param[in] values_to_replace_end  Device pointer to the end of the sequence
 * of old values to be replaced
 * @param[in] d_replacement_values Device array with the new values
 * @param[in] replacement_valid Valid mask associated with d_replacement_values
 *
 * @returns
 */
/* ----------------------------------------------------------------------------*/
template <class T, bool input_has_nulls, bool replacement_has_nulls>
__global__ void replace_kernel(cudf::column_device_view input,
                               cudf::mutable_column_device_view output,
                               cudf::size_type* __restrict__ output_valid_count,
                               cudf::size_type nrows,
                               cudf::column_device_view values_to_replace,
                               cudf::column_device_view replacement)
{
  T* __restrict__ output_data = output.data<T>();

  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t active_mask = 0xffffffff;
  active_mask          = __ballot_sync(active_mask, i < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (i < nrows) {
    bool output_is_valid{true};
    bool input_is_valid{true};
    if (input_has_nulls) {
      input_is_valid  = input.is_valid_nocheck(i);
      output_is_valid = input_is_valid;
    }
    if (input_is_valid)
      thrust::tie(output_data[i], output_is_valid) = get_new_value<T, replacement_has_nulls>(
        i,
        input.data<T>(),
        values_to_replace.data<T>(),
        values_to_replace.data<T>() + values_to_replace.size(),
        replacement.data<T>(),
        replacement.null_mask());

    /* output valid counts calculations*/
    if (input_has_nulls or replacement_has_nulls) {
      uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
      if (0 == lane_id) {
        output.set_mask_word(cudf::word_index(i), bitmask);
        valid_sum += __popc(bitmask);
      }
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }
  if (input_has_nulls or replacement_has_nulls) {
    // Compute total valid count for this block and add it to global count
    uint32_t block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
    // one thread computes and adds to output_valid_count
    if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
  }
}

/*
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_kernel` with the appropriate data types.
 */
struct replace_kernel_forwarder {
  template <typename col_type, std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input_col,
                                           cudf::column_view const& values_to_replace,
                                           cudf::column_view const& replacement_values,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
    cudf::size_type* valid_count = valid_counter.data();

    auto replace = replace_kernel<col_type, true, true>;
    if (input_col.has_nulls()) {
      if (replacement_values.has_nulls()) {
        replace = replace_kernel<col_type, true, true>;
      } else {
        replace = replace_kernel<col_type, true, false>;
      }
    } else {
      if (replacement_values.has_nulls()) {
        replace = replace_kernel<col_type, false, true>;
      } else {
        replace = replace_kernel<col_type, false, false>;
      }
    }

    std::unique_ptr<cudf::column> output;
    if (input_col.has_nulls() || replacement_values.has_nulls()) {
      output = cudf::detail::allocate_like(
        input_col, input_col.size(), cudf::mask_allocation_policy::ALWAYS, mr, stream);
    } else {
      output = cudf::detail::allocate_like(
        input_col, input_col.size(), cudf::mask_allocation_policy::NEVER, mr, stream);
    }

    cudf::mutable_column_view outputView = output->mutable_view();

    cudf::detail::grid_1d grid{outputView.size(), BLOCK_SIZE, 1};

    auto device_in                 = cudf::column_device_view::create(input_col);
    auto device_out                = cudf::mutable_column_device_view::create(outputView);
    auto device_values_to_replace  = cudf::column_device_view::create(values_to_replace);
    auto device_replacement_values = cudf::column_device_view::create(replacement_values);

    replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(*device_in,
                                                        *device_out,
                                                        valid_count,
                                                        outputView.size(),
                                                        *device_values_to_replace,
                                                        *device_replacement_values);

    if (outputView.nullable()) {
      output->set_null_count(output->size() - valid_counter.value(stream));
    }
    return output;
  }

  template <typename col_type, std::enable_if_t<not cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input_col,
                                           cudf::column_view const& values_to_replace,
                                           cudf::column_view const& replacement_values,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    CUDF_FAIL("No specialization exists for this type");
  }
};

template <>
std::unique_ptr<cudf::column> replace_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input_col,
  cudf::column_view const& values_to_replace,
  cudf::column_view const& replacement_values,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
  cudf::size_type* valid_count = valid_counter.data();

  auto replace_first  = replace_strings_first_pass<true, false>;
  auto replace_second = replace_strings_second_pass<true, false>;
  if (input_col.has_nulls()) {
    if (replacement_values.has_nulls()) {
      replace_first  = replace_strings_first_pass<true, true>;
      replace_second = replace_strings_second_pass<true, true>;
    }
  } else {
    if (replacement_values.has_nulls()) {
      replace_first  = replace_strings_first_pass<false, true>;
      replace_second = replace_strings_second_pass<false, true>;
    } else {
      replace_first  = replace_strings_first_pass<false, false>;
      replace_second = replace_strings_second_pass<false, false>;
    }
  }

  // Create new offsets column to use in kernel
  std::unique_ptr<cudf::column> sizes = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT32), input_col.size(), cudf::mask_state::UNALLOCATED, stream);
  std::unique_ptr<cudf::column> indices = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT32), input_col.size(), cudf::mask_state::UNALLOCATED, stream);

  auto sizes_view   = sizes->mutable_view();
  auto indices_view = indices->mutable_view();

  auto device_in                = cudf::column_device_view::create(input_col);
  auto device_values_to_replace = cudf::column_device_view::create(values_to_replace);
  auto device_replacement       = cudf::column_device_view::create(replacement_values);
  auto device_sizes             = cudf::mutable_column_device_view::create(sizes_view);
  auto device_indices           = cudf::mutable_column_device_view::create(indices_view);

  rmm::device_buffer valid_bits =
    cudf::create_null_mask(input_col.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // Call first pass kernel to get sizes in offsets
  cudf::detail::grid_1d grid{input_col.size(), BLOCK_SIZE, 1};
  replace_first<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
    *device_in,
    *device_values_to_replace,
    *device_replacement,
    *device_sizes,
    *device_indices,
    reinterpret_cast<cudf::bitmask_type*>(valid_bits.data()),
    valid_count);

  std::unique_ptr<cudf::column> offsets = cudf::strings::detail::make_offsets_child_column(
    sizes_view.begin<int32_t>(), sizes_view.end<int32_t>(), mr, stream);
  auto offsets_view   = offsets->mutable_view();
  auto device_offsets = cudf::mutable_column_device_view::create(offsets_view);
  int32_t size;
  CUDA_TRY(cudaMemcpyAsync(
    &size, offsets_view.end<int32_t>() - 1, sizeof(int32_t), cudaMemcpyDefault, stream));

  // Allocate chars array and output null mask
  cudf::size_type null_count                 = input_col.size() - valid_counter.value(stream);
  std::unique_ptr<cudf::column> output_chars = cudf::strings::detail::create_chars_child_column(
    input_col.size(), null_count, size, mr, stream);

  auto output_chars_view = output_chars->mutable_view();
  auto device_chars      = cudf::mutable_column_device_view::create(output_chars_view);

  replace_second<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
    *device_in, *device_replacement, *device_offsets, *device_chars, *device_indices);

  return cudf::make_strings_column(input_col.size(),
                                   std::move(offsets),
                                   std::move(output_chars),
                                   null_count,
                                   std::move(valid_bits),
                                   stream,
                                   mr);
}

}  // end anonymous namespace

namespace cudf {
namespace detail {
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   rmm::mr::device_memory_resource* mr,
                                                   cudaStream_t stream)
{
  CUDF_EXPECTS(values_to_replace.size() == replacement_values.size(),
               "values_to_replace and replacement_values size mismatch.");

  CUDF_EXPECTS(
    input_col.type() == values_to_replace.type() && input_col.type() == replacement_values.type(),
    "Columns type mismatch");
  CUDF_EXPECTS(values_to_replace.has_nulls() == false, "values_to_replace must not have nulls");

  if (0 == input_col.size() || 0 == values_to_replace.size() || 0 == replacement_values.size()) {
    return std::make_unique<cudf::column>(input_col);
  }

  return cudf::type_dispatcher(input_col.type(),
                               replace_kernel_forwarder{},
                               input_col,
                               values_to_replace,
                               replacement_values,
                               mr,
                               stream);
}

}  // namespace detail

/* --------------------------------------------------------------------------*/
/*
 * @brief Replace elements from `input_col` according to the mapping `values_to_replace` to
 *        `replacement_values`, that is, replace all `values_to_replace[i]` present in `input_col`
 *        with `replacement_values[i]`.
 *
 * @param[in] col column_view of the data to be modified
 * @param[in] values_to_replace column_view of the old values to be replaced
 * @param[in] replacement_values column_view of the new values
 *
 * @returns output cudf::column with the modified data
 */
/* ----------------------------------------------------------------------------*/
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   rmm::mr::device_memory_resource* mr)
{
  return cudf::detail::find_and_replace_all(
    input_col, values_to_replace, replacement_values, mr, 0);
}
}  // namespace cudf

namespace {  // anonymous

template <int phase, bool replacement_has_nulls>
__global__ void replace_nulls_strings(cudf::column_device_view input,
                                      cudf::column_device_view replacement,
                                      cudf::bitmask_type* output_valid,
                                      cudf::size_type* offsets,
                                      char* chars,
                                      cudf::size_type* valid_counter)
{
  cudf::size_type nrows = input.size();
  cudf::size_type i     = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t active_mask = 0xffffffff;
  active_mask          = __ballot_sync(active_mask, i < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (i < nrows) {
    bool input_is_valid  = input.is_valid_nocheck(i);
    bool output_is_valid = true;

    if (replacement_has_nulls && !input_is_valid) {
      output_is_valid = replacement.is_valid_nocheck(i);
    }

    cudf::string_view out;
    if (input_is_valid) {
      out = input.element<cudf::string_view>(i);
    } else if (output_is_valid) {
      out = replacement.element<cudf::string_view>(i);
    }

    bool nonzero_output = (input_is_valid || output_is_valid);

    if (phase == 0) {
      offsets[i]       = nonzero_output ? out.size_bytes() : 0;
      uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
      if (0 == lane_id) {
        output_valid[cudf::word_index(i)] = bitmask;
        valid_sum += __popc(bitmask);
      }
    } else if (phase == 1) {
      if (nonzero_output) std::memcpy(chars + offsets[i], out.data(), out.size_bytes());
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }

  // Compute total valid count for this block and add it to global count
  uint32_t block_valid_count = cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
  // one thread computes and adds to output_valid_count
  if (threadIdx.x == 0) { atomicAdd(valid_counter, block_valid_count); }
}

template <typename Type, bool replacement_has_nulls>
__global__ void replace_nulls(cudf::column_device_view input,
                              cudf::column_device_view replacement,
                              cudf::mutable_column_device_view output,
                              cudf::size_type* output_valid_count)
{
  cudf::size_type nrows = input.size();
  cudf::size_type i     = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t active_mask = 0xffffffff;
  active_mask          = __ballot_sync(active_mask, i < nrows);
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  uint32_t valid_sum{0};

  while (i < nrows) {
    bool input_is_valid  = input.is_valid_nocheck(i);
    bool output_is_valid = true;
    if (input_is_valid) {
      output.data<Type>()[i] = input.element<Type>(i);
    } else {
      if (replacement_has_nulls) { output_is_valid = replacement.is_valid_nocheck(i); }
      output.data<Type>()[i] = replacement.element<Type>(i);
    }

    /* output valid counts calculations*/
    if (replacement_has_nulls) {
      uint32_t bitmask = __ballot_sync(active_mask, output_is_valid);
      if (0 == lane_id) {
        output.set_mask_word(cudf::word_index(i), bitmask);
        valid_sum += __popc(bitmask);
      }
    }

    i += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }
  if (replacement_has_nulls) {
    // Compute total valid count for this block and add it to global count
    uint32_t block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
    // one thread computes and adds to output_valid_count
    if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
  }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the appropriate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_column_kernel_forwarder {
  template <typename col_type, std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::column_view const& replacement,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    cudf::size_type nrows = input.size();
    cudf::detail::grid_1d grid{nrows, BLOCK_SIZE};

    std::unique_ptr<cudf::column> output;
    if (replacement.has_nulls())
      output = cudf::detail::allocate_like(
        input, input.size(), cudf::mask_allocation_policy::ALWAYS, mr, stream);
    else
      output = cudf::detail::allocate_like(
        input, input.size(), cudf::mask_allocation_policy::NEVER, mr, stream);
    auto output_view = output->mutable_view();

    auto replace = replace_nulls<col_type, false>;
    if (output_view.nullable()) replace = replace_nulls<col_type, true>;

    auto device_in          = cudf::column_device_view::create(input);
    auto device_out         = cudf::mutable_column_device_view::create(output_view);
    auto device_replacement = cudf::column_device_view::create(replacement);

    rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
    cudf::size_type* valid_count = valid_counter.data();

    replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
      *device_in, *device_replacement, *device_out, valid_count);

    if (output_view.nullable()) {
      output->set_null_count(output->size() - valid_counter.value(stream));
    }

    return output;
  }

  template <typename col_type, std::enable_if_t<not cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::column_view const& replacement,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    CUDF_FAIL("No specialization exists for the given type.");
  }
};

template <>
std::unique_ptr<cudf::column> replace_nulls_column_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input,
  cudf::column_view const& replacement,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
  cudf::size_type* valid_count = valid_counter.data();

  auto replace_first  = replace_nulls_strings<0, false>;
  auto replace_second = replace_nulls_strings<1, false>;
  if (replacement.has_nulls()) {
    replace_first  = replace_nulls_strings<0, true>;
    replace_second = replace_nulls_strings<1, true>;
  }

  // Create new offsets column to use in kernel
  std::unique_ptr<cudf::column> sizes = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT32), input.size(), cudf::mask_state::UNALLOCATED, stream);

  auto sizes_view         = sizes->mutable_view();
  auto device_in          = cudf::column_device_view::create(input);
  auto device_replacement = cudf::column_device_view::create(replacement);

  rmm::device_buffer valid_bits =
    cudf::create_null_mask(input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // Call first pass kernel to get sizes in offsets
  cudf::detail::grid_1d grid{input.size(), BLOCK_SIZE, 1};
  replace_first<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
    *device_in,
    *device_replacement,
    reinterpret_cast<cudf::bitmask_type*>(valid_bits.data()),
    sizes_view.begin<cudf::size_type>(),
    nullptr,
    valid_count);

  std::unique_ptr<cudf::column> offsets = cudf::strings::detail::make_offsets_child_column(
    sizes_view.begin<int32_t>(), sizes_view.end<int32_t>(), mr, stream);
  auto offsets_view = offsets->mutable_view();

  int32_t size;
  CUDA_TRY(cudaMemcpyAsync(
    &size, offsets_view.end<int32_t>() - 1, sizeof(int32_t), cudaMemcpyDefault, stream));

  // Allocate chars array and output null mask
  cudf::size_type null_count = input.size() - valid_counter.value(stream);
  std::unique_ptr<cudf::column> output_chars =
    cudf::strings::detail::create_chars_child_column(input.size(), null_count, size, mr, stream);

  auto output_chars_view = output_chars->mutable_view();

  replace_second<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(
    *device_in,
    *device_replacement,
    reinterpret_cast<cudf::bitmask_type*>(valid_bits.data()),
    offsets_view.begin<cudf::size_type>(),
    output_chars_view.data<char>(),
    valid_count);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   std::move(output_chars),
                                   input.size() - valid_counter.value(stream),
                                   std::move(valid_bits),
                                   stream,
                                   mr);
}

template <typename T>
struct replace_nulls_functor {
  T* value_it;
  replace_nulls_functor(T* _value_it) : value_it(_value_it) {}
  __device__ T operator()(T input, bool is_valid) { return is_valid ? input : *value_it; }
};

/* --------------------------------------------------------------------------*/
/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the appropriate data types.
 */
/* ----------------------------------------------------------------------------*/
struct replace_nulls_scalar_kernel_forwarder {
  template <typename col_type, std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::scalar const& replacement,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    std::unique_ptr<cudf::column> output =
      cudf::allocate_like(input, cudf::mask_allocation_policy::NEVER, mr);
    auto output_view = output->mutable_view();

    using ScalarType = cudf::scalar_type_t<col_type>;
    auto s1          = static_cast<ScalarType const&>(replacement);
    auto device_in   = cudf::column_device_view::create(input);

    replace_nulls_functor<col_type> func(s1.data());
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.data<col_type>(),
                      input.data<col_type>() + input.size(),
                      cudf::detail::make_validity_iterator(*device_in),
                      output_view.data<col_type>(),
                      func);
    return output;
  }

  template <typename col_type, std::enable_if_t<not cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::scalar const& replacement,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    CUDF_FAIL("No specialization exists for the given type.");
  }
};

template <>
std::unique_ptr<cudf::column> replace_nulls_scalar_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input,
  cudf::scalar const& replacement,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  cudf::strings_column_view input_s(input);
  const cudf::string_scalar& repl = static_cast<const cudf::string_scalar&>(replacement);
  return cudf::strings::replace_nulls(input_s, repl, mr);
}

}  // end anonymous namespace

namespace cudf {
namespace detail {
std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream)
{
  CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
  CUDF_EXPECTS(replacement.size() == input.size(), "Column size mismatch");

  if (input.size() == 0) { return cudf::empty_like(input); }

  if (!input.has_nulls()) { return std::make_unique<cudf::column>(input); }

  return cudf::type_dispatcher(
    input.type(), replace_nulls_column_kernel_forwarder{}, input, replacement, mr, stream);
}

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::scalar const& replacement,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream)
{
  if (input.size() == 0) { return cudf::empty_like(input); }

  if (!input.has_nulls() || !replacement.is_valid()) {
    return std::make_unique<cudf::column>(input, stream, mr);
  }

  CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");

  return cudf::type_dispatcher(
    input.type(), replace_nulls_scalar_kernel_forwarder{}, input, replacement, mr, stream);
}

}  // namespace detail

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::replace_nulls(input, replacement, mr, 0);
}

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::scalar const& replacement,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::replace_nulls(input, replacement, mr, 0);
}
}  // namespace cudf

namespace cudf {
namespace detail {
namespace {

struct replace_nans_functor {
  template <typename T, typename Replacement>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input,
    Replacement const& replacement,
    bool replacement_nullable,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    CUDF_EXPECTS(input.type() == replacement.type(),
                 "Input and replacement must be of the same type");

    if (input.size() == 0) { return cudf::make_empty_column(input.type()); }

    auto input_device_view = column_device_view::create(input);
    size_type size         = input.size();

    auto predicate = [dinput = *input_device_view] __device__(auto i) {
      return dinput.is_null(i) or !std::isnan(dinput.element<T>(i));
    };

    if (input.has_nulls()) {
      auto input_pair_iterator = make_pair_iterator<T, true>(*input_device_view);
      if (replacement_nullable) {
        auto replacement_pair_iterator = make_pair_iterator<T, true>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            mr,
                            stream);
      } else {
        auto replacement_pair_iterator = make_pair_iterator<T, false>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            mr,
                            stream);
      }
    } else {
      auto input_pair_iterator = make_pair_iterator<T, false>(*input_device_view);
      if (replacement_nullable) {
        auto replacement_pair_iterator = make_pair_iterator<T, true>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            mr,
                            stream);
      } else {
        auto replacement_pair_iterator = make_pair_iterator<T, false>(replacement);
        return copy_if_else(false,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            mr,
                            stream);
      }
    }
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

}  // namespace
std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == replacement.size(),
               "Input and replacement must be of the same size");

  return type_dispatcher(input.type(),
                         replace_nans_functor{},
                         input,
                         *column_device_view::create(replacement),
                         replacement.nullable(),
                         mr,
                         stream);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input.type(), replace_nans_functor{}, input, replacement, true, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, 0, mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, 0, mr);
}

}  // namespace cudf

namespace {  // anonymous

template <typename T>
struct normalize_nans_and_zeros_lambda {
  cudf::column_device_view in;
  T __device__ operator()(cudf::size_type i)
  {
    auto e = in.element<T>(i);
    if (isnan(e)) { return std::numeric_limits<T>::quiet_NaN(); }
    if (T{0.0} == e) { return T{0.0}; }
    return e;
  }
};

/* --------------------------------------------------------------------------*/
/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `normalize_nans_and_zeros` with the appropriate data types.
 */
/* ----------------------------------------------------------------------------*/
struct normalize_nans_and_zeros_kernel_forwarder {
  // floats and doubles. what we really care about.
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  cudaStream_t stream)
  {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in.size()),
                      out.head<T>(),
                      normalize_nans_and_zeros_lambda<T>{in});
  }

  // if we get in here for anything but a float or double, that's a problem.
  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  cudaStream_t stream)
  {
    CUDF_FAIL("Unexpected non floating-point type.");
  }
};

}  // end anonymous namespace

namespace cudf {
namespace detail {
void normalize_nans_and_zeros(mutable_column_view in_out, cudaStream_t stream = 0)
{
  if (in_out.size() == 0) { return; }
  CUDF_EXPECTS(
    in_out.type() == data_type(type_id::FLOAT32) || in_out.type() == data_type(type_id::FLOAT64),
    "Expects float or double input");

  // wrapping the in_out data in a column_view so we can call the same lower level code.
  // that we use for the non in-place version.
  column_view input = in_out;

  // to device. unique_ptr which gets automatically cleaned up when we leave
  auto device_in = column_device_view::create(input);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto device_out = mutable_column_device_view::create(in_out);

  // invoke the actual kernel.
  cudf::type_dispatcher(
    input.type(), normalize_nans_and_zeros_kernel_forwarder{}, *device_in, *device_out, stream);
}

}  // namespace detail

/*
 * @brief Makes all NaNs and zeroes positive.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] column_view representing input data
 * @param[in] device_memory_resource allocator for allocating output data
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // output. copies the input
  std::unique_ptr<column> out = std::make_unique<column>(input, (cudaStream_t)0, mr);
  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto out_view = out->mutable_view();

  detail::normalize_nans_and_zeros(out_view, 0);

  return out;
}

/*
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] mutable_column_view representing input data. data is processed in-place
 */
void normalize_nans_and_zeros(mutable_column_view& in_out)
{
  CUDF_FUNC_RANGE();
  detail::normalize_nans_and_zeros(in_out, 0);
}

}  // namespace cudf
