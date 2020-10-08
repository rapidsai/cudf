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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/transform.h>

namespace {  // anonymous

static constexpr int BLOCK_SIZE = 256;

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

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the appropriate data types.
 */
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

template <>
std::unique_ptr<cudf::column> replace_nulls_column_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input,
  cudf::column_view const& replacement,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  cudf::dictionary_column_view dict_input(input);
  cudf::dictionary_column_view dict_repl(replacement);
  return cudf::dictionary::detail::replace_nulls(dict_input, dict_repl, mr, stream);
}

template <typename T>
struct replace_nulls_functor {
  T* value_it;
  replace_nulls_functor(T* _value_it) : value_it(_value_it) {}
  __device__ T operator()(T input, bool is_valid) { return is_valid ? input : *value_it; }
};

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the appropriate data types.
 */
struct replace_nulls_scalar_kernel_forwarder {
  template <typename col_type,
            typename std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::scalar const& replacement,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream = 0)
  {
    CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
    std::unique_ptr<cudf::column> output =
      cudf::allocate_like(input, cudf::mask_allocation_policy::NEVER, mr);
    auto output_view = output->mutable_view();

    using Type       = cudf::device_storage_type_t<col_type>;
    using ScalarType = cudf::scalar_type_t<col_type>;
    auto s1          = static_cast<ScalarType const&>(replacement);
    auto device_in   = cudf::column_device_view::create(input);

    auto func = replace_nulls_functor<Type>{s1.data()};
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.data<Type>(),
                      input.data<Type>() + input.size(),
                      cudf::detail::make_validity_iterator(*device_in),
                      output_view.data<Type>(),
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
  CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
  cudf::strings_column_view input_s(input);
  const cudf::string_scalar& repl = static_cast<const cudf::string_scalar&>(replacement);
  return cudf::strings::replace_nulls(input_s, repl, mr);
}

template <>
std::unique_ptr<cudf::column> replace_nulls_scalar_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input,
  cudf::scalar const& replacement,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  cudf::dictionary_column_view dict_input(input);
  return cudf::dictionary::detail::replace_nulls(dict_input, replacement, mr, stream);
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
