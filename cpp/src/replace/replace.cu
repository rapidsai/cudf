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
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/find.h>
#include <rmm/device_scalar.hpp>

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
  T new_value{};
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

/**
 * @brief Kernel which does the first pass of strings replace.
 *
 * It computes the output null_mask, null_count, and the offsets.
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

/**
 * @brief Kernel which does the second pass of strings replace.
 *
 * It copies the string data needed from input and replacement into the new strings column chars
 * column.
 *
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
 */
template <class T, bool input_has_nulls, bool replacement_has_nulls>
__global__ void replace_kernel(cudf::column_device_view input,
                               cudf::mutable_column_device_view output,
                               cudf::size_type* __restrict__ output_valid_count,
                               cudf::size_type nrows,
                               cudf::column_device_view values_to_replace,
                               cudf::column_device_view replacement)
{
  using Type = cudf::device_storage_type_t<T>;

  Type* __restrict__ output_data = output.data<Type>();

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
      thrust::tie(output_data[i], output_is_valid) = get_new_value<Type, replacement_has_nulls>(
        i,
        input.data<Type>(),
        values_to_replace.data<Type>(),
        values_to_replace.data<Type>() + values_to_replace.size(),
        replacement.data<Type>(),
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

/**
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

    auto replace = [&] {
      if (input_col.has_nulls())
        return replacement_values.has_nulls() ? replace_kernel<col_type, true, true>
                                              : replace_kernel<col_type, true, false>;
      else
        return replacement_values.has_nulls() ? replace_kernel<col_type, false, true>
                                              : replace_kernel<col_type, false, false>;
    }();

    auto output = [&] {
      auto const mask_allocation_policy = input_col.has_nulls() || replacement_values.has_nulls()
                                            ? cudf::mask_allocation_policy::ALWAYS
                                            : cudf::mask_allocation_policy::NEVER;
      return cudf::detail::allocate_like(
        input_col, input_col.size(), mask_allocation_policy, mr, stream);
    }();

    auto output_view = output->mutable_view();
    auto grid        = cudf::detail::grid_1d{output_view.size(), BLOCK_SIZE, 1};

    auto device_in                 = cudf::column_device_view::create(input_col);
    auto device_out                = cudf::mutable_column_device_view::create(output_view);
    auto device_values_to_replace  = cudf::column_device_view::create(values_to_replace);
    auto device_replacement_values = cudf::column_device_view::create(replacement_values);

    replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(*device_in,
                                                        *device_out,
                                                        valid_count,
                                                        output_view.size(),
                                                        *device_values_to_replace,
                                                        *device_replacement_values);

    if (output_view.nullable()) {
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

template <>
std::unique_ptr<cudf::column> replace_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input_col,
  cudf::column_view const& values_to_replace,
  cudf::column_view const& replacement_values,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto input        = cudf::dictionary_column_view(input_col);
  auto values       = cudf::dictionary_column_view(values_to_replace);
  auto replacements = cudf::dictionary_column_view(replacement_values);

  auto matched_input = [&] {
    auto new_keys = cudf::detail::concatenate(
      {values.keys(), replacements.keys()}, rmm::mr::get_current_device_resource(), stream);
    return cudf::dictionary::detail::add_keys(input, new_keys->view(), mr, stream);
  }();
  auto matched_view   = cudf::dictionary_column_view(matched_input->view());
  auto matched_values = cudf::dictionary::detail::set_keys(
    values, matched_view.keys(), rmm::mr::get_current_device_resource(), stream);
  auto matched_replacements = cudf::dictionary::detail::set_keys(
    replacements, matched_view.keys(), rmm::mr::get_current_device_resource(), stream);

  auto indices_type = matched_view.indices().type();
  auto new_indices  = cudf::type_dispatcher(
    indices_type,
    replace_kernel_forwarder{},
    matched_view.get_indices_annotated(),
    cudf::dictionary_column_view(matched_values->view()).indices(),
    cudf::dictionary_column_view(matched_replacements->view()).get_indices_annotated(),
    mr,
    stream);
  auto null_count     = new_indices->null_count();
  auto contents       = new_indices->release();
  auto indices_column = std::make_unique<cudf::column>(
    indices_type, input.size(), *(contents.data.release()), rmm::device_buffer{0, stream, mr}, 0);
  std::unique_ptr<cudf::column> keys_column(std::move(matched_input->release().children.back()));
  return cudf::make_dictionary_column(std::move(keys_column),
                                      std::move(indices_column),
                                      std::move(*(contents.null_mask.release())),
                                      null_count);
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

/**
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
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   rmm::mr::device_memory_resource* mr)
{
  return cudf::detail::find_and_replace_all(
    input_col, values_to_replace, replacement_values, mr, 0);
}
}  // namespace cudf
