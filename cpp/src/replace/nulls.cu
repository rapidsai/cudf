/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/replace/nulls.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace {  // anonymous

static constexpr int BLOCK_SIZE = 256;

template <int phase, bool replacement_has_nulls>
CUDF_KERNEL void replace_nulls_strings(cudf::column_device_view input,
                                       cudf::column_device_view replacement,
                                       cudf::bitmask_type* output_valid,
                                       cudf::size_type* offsets,
                                       char* chars,
                                       cudf::size_type* valid_counter)
{
  cudf::size_type nrows = input.size();
  auto i                = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = cudf::detail::grid_1d::grid_stride();

  uint32_t active_mask = 0xffff'ffff;
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

    i += stride;
    active_mask = __ballot_sync(active_mask, i < nrows);
  }

  // Compute total valid count for this block and add it to global count
  uint32_t block_valid_count = cudf::detail::single_lane_block_sum_reduce<BLOCK_SIZE, 0>(valid_sum);
  // one thread computes and adds to output_valid_count
  if (threadIdx.x == 0) { atomicAdd(valid_counter, block_valid_count); }
}

template <typename Type, bool replacement_has_nulls>
CUDF_KERNEL void replace_nulls(cudf::column_device_view input,
                               cudf::column_device_view replacement,
                               cudf::mutable_column_device_view output,
                               cudf::size_type* output_valid_count)
{
  cudf::size_type nrows = input.size();
  auto i                = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = cudf::detail::grid_1d::grid_stride();

  uint32_t active_mask = 0xffff'ffff;
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

    i += stride;
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
  template <typename col_type, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<col_type>())>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::column_view const& replacement,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    cudf::size_type nrows = input.size();
    cudf::detail::grid_1d grid{nrows, BLOCK_SIZE};

    auto output =
      cudf::detail::allocate_like(input,
                                  input.size(),
                                  replacement.has_nulls() ? cudf::mask_allocation_policy::ALWAYS
                                                          : cudf::mask_allocation_policy::NEVER,
                                  stream,
                                  mr);

    auto output_view = output->mutable_view();

    auto replace = replace_nulls<col_type, false>;
    if (output_view.nullable()) replace = replace_nulls<col_type, true>;

    auto device_in          = cudf::column_device_view::create(input, stream);
    auto device_out         = cudf::mutable_column_device_view::create(output_view, stream);
    auto device_replacement = cudf::column_device_view::create(replacement, stream);

    rmm::device_scalar<cudf::size_type> valid_counter(0, stream);
    cudf::size_type* valid_count = valid_counter.data();

    replace<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
      *device_in, *device_replacement, *device_out, valid_count);

    if (output_view.nullable()) {
      output->set_null_count(output->size() - valid_counter.value(stream));
    }

    return output;
  }

  template <typename col_type, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<col_type>())>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("No specialization exists for the given type.");
  }
};

template <>
std::unique_ptr<cudf::column> replace_nulls_column_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input,
  cudf::column_view const& replacement,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
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
  auto device_in          = cudf::column_device_view::create(input, stream);
  auto device_replacement = cudf::column_device_view::create(replacement, stream);

  rmm::device_buffer valid_bits =
    cudf::detail::create_null_mask(input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // Call first pass kernel to get sizes in offsets
  cudf::detail::grid_1d grid{input.size(), BLOCK_SIZE, 1};
  replace_first<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    *device_in,
    *device_replacement,
    reinterpret_cast<cudf::bitmask_type*>(valid_bits.data()),
    sizes_view.begin<cudf::size_type>(),
    nullptr,
    valid_count);

  auto [offsets, bytes] = cudf::detail::make_offsets_child_column(
    sizes_view.begin<int32_t>(), sizes_view.end<int32_t>(), stream, mr);

  auto offsets_view = offsets->mutable_view();

  // Allocate chars array and output null mask
  std::unique_ptr<cudf::column> output_chars =
    cudf::strings::detail::create_chars_child_column(bytes, stream, mr);

  auto output_chars_view = output_chars->mutable_view();

  replace_second<<<grid.num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
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
                                   std::move(valid_bits));
}

template <>
std::unique_ptr<cudf::column> replace_nulls_column_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input,
  cudf::column_view const& replacement,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  cudf::dictionary_column_view dict_input(input);
  cudf::dictionary_column_view dict_repl(replacement);
  return cudf::dictionary::detail::replace_nulls(dict_input, dict_repl, stream, mr);
}

template <typename T>
struct replace_nulls_functor {
  T const* value_it;
  replace_nulls_functor(T const* _value_it) : value_it(_value_it) {}
  __device__ T operator()(T input, bool is_valid) { return is_valid ? input : *value_it; }
};

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `replace_nulls` with the appropriate data types.
 */
struct replace_nulls_scalar_kernel_forwarder {
  template <typename col_type, std::enable_if_t<cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::scalar const& replacement,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
    std::unique_ptr<cudf::column> output = cudf::detail::allocate_like(
      input, input.size(), cudf::mask_allocation_policy::NEVER, stream, mr);
    auto output_view = output->mutable_view();

    using ScalarType = cudf::scalar_type_t<col_type>;
    auto& s1         = static_cast<ScalarType const&>(replacement);
    auto device_in   = cudf::column_device_view::create(input, stream);

    auto func = replace_nulls_functor<col_type>{s1.data()};
    thrust::transform(rmm::exec_policy(stream),
                      input.data<col_type>(),
                      input.data<col_type>() + input.size(),
                      cudf::detail::make_validity_iterator(*device_in),
                      output_view.data<col_type>(),
                      func);
    return output;
  }

  template <typename col_type, std::enable_if_t<not cudf::is_fixed_width<col_type>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           cudf::scalar const&,
                                           rmm::cuda_stream_view,
                                           rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("No specialization exists for the given type.");
  }
};

template <>
std::unique_ptr<cudf::column> replace_nulls_scalar_kernel_forwarder::operator()<cudf::string_view>(
  cudf::column_view const& input,
  cudf::scalar const& replacement,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
  cudf::strings_column_view input_s(input);
  cudf::string_scalar const& repl = static_cast<cudf::string_scalar const&>(replacement);
  return cudf::strings::detail::replace_nulls(input_s, repl, stream, mr);
}

template <>
std::unique_ptr<cudf::column> replace_nulls_scalar_kernel_forwarder::operator()<cudf::dictionary32>(
  cudf::column_view const& input,
  cudf::scalar const& replacement,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  cudf::dictionary_column_view dict_input(input);
  return cudf::dictionary::detail::replace_nulls(dict_input, replacement, stream, mr);
}

/**
 * @brief Function used by replace_nulls policy
 */

std::unique_ptr<cudf::column> replace_nulls_policy_impl(cudf::column_view const& input,
                                                        cudf::replace_policy const& replace_policy,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource* mr)
{
  auto device_in = cudf::column_device_view::create(input, stream);
  auto index     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto valid_it  = cudf::detail::make_validity_iterator(*device_in);
  auto in_begin  = thrust::make_zip_iterator(thrust::make_tuple(index, valid_it));

  rmm::device_uvector<cudf::size_type> gather_map(input.size(), stream);
  auto gm_begin = thrust::make_zip_iterator(
    thrust::make_tuple(gather_map.begin(), thrust::make_discard_iterator()));

  auto func = cudf::detail::replace_policy_functor();
  if (replace_policy == cudf::replace_policy::PRECEDING) {
    thrust::inclusive_scan(
      rmm::exec_policy(stream), in_begin, in_begin + input.size(), gm_begin, func);
  } else {
    auto in_rbegin = thrust::make_reverse_iterator(in_begin + input.size());
    auto gm_rbegin = thrust::make_reverse_iterator(gm_begin + gather_map.size());
    thrust::inclusive_scan(
      rmm::exec_policy(stream), in_rbegin, in_rbegin + input.size(), gm_rbegin, func);
  }

  auto output = cudf::detail::gather(cudf::table_view({input}),
                                     gather_map,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::detail::negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);

  return std::move(output->release()[0]);
}

}  // end anonymous namespace

namespace cudf {
namespace detail {

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == replacement.type(), "Data type mismatch");
  CUDF_EXPECTS(replacement.size() == input.size(), "Column size mismatch");

  if (input.is_empty()) { return cudf::empty_like(input); }
  if (!input.has_nulls()) { return std::make_unique<cudf::column>(input, stream, mr); }

  return cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), replace_nulls_column_kernel_forwarder{}, input, replacement, stream, mr);
}

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::scalar const& replacement,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return cudf::empty_like(input); }
  if (!input.has_nulls() || !replacement.is_valid(stream)) {
    return std::make_unique<cudf::column>(input, stream, mr);
  }

  return cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), replace_nulls_scalar_kernel_forwarder{}, input, replacement, stream, mr);
}

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::replace_policy const& replace_policy,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return cudf::empty_like(input); }
  if (!input.has_nulls()) { return std::make_unique<cudf::column>(input, stream, mr); }

  return replace_nulls_policy_impl(input, replace_policy, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nulls(input, replacement, stream, mr);
}

std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::scalar const& replacement,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nulls(input, replacement, stream, mr);
}

std::unique_ptr<cudf::column> replace_nulls(column_view const& input,
                                            replace_policy const& replace_policy,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nulls(input, replace_policy, stream, mr);
}

}  // namespace cudf
