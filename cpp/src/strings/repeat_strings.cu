/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<string_scalar> repeat_string(string_scalar const& input,
                                             size_type repeat_times,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (!input.is_valid(stream)) { return std::make_unique<string_scalar>("", false, stream, mr); }
  if (input.size() == 0 || repeat_times <= 0) {
    return std::make_unique<string_scalar>("", true, stream, mr);
  }
  if (repeat_times == 1) { return std::make_unique<string_scalar>(input, stream, mr); }

  CUDF_EXPECTS(input.size() <= std::numeric_limits<size_type>::max() / repeat_times,
               "The output string has size that exceeds the maximum allowed size.");

  auto const str_size = input.size();
  auto const iter     = thrust::make_counting_iterator(0);
  auto buff           = rmm::device_buffer(repeat_times * input.size(), stream, mr);

  // Pull data from the input string into each byte of the output string.
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + repeat_times * str_size,
                    static_cast<char*>(buff.data()),
                    [in_ptr = input.data(), str_size] __device__(const auto idx) {
                      return in_ptr[idx % str_size];
                    });

  return std::make_unique<string_scalar>(std::move(buff));
}

namespace {
/**
 * @brief Generate a strings column in which each row is an empty or null string.
 *
 * The output strings column has the same bitmask as the input column.
 */
auto generate_empty_output(strings_column_view const& input,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  auto chars_column = create_chars_child_column(strings_count, 0, stream, mr);

  auto offsets_column = make_numeric_column(
    data_type{type_to_id<offset_type>()}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  CUDA_TRY(cudaMemsetAsync(offsets_column->mutable_view().template data<offset_type>(),
                           0,
                           offsets_column->size() * sizeof(offset_type),
                           stream.value()));

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
}

/**
 * @brief Functor to compute string sizes and repeat the input strings.
 *
 * This functor is called only when `repeat_times > 0`. In addition, the total number of threads
 * running this functor is `repeat_times * strings_count` (instead of `string_count`) for maximizing
 * parallelism and better load-balancing.
 */
struct compute_size_and_repeat_fn {
  column_device_view const strings_dv;
  size_type const repeat_times;
  bool const has_nulls;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only repeat strings.
  char* d_chars{nullptr};

  // `idx` will be in the range of [0, repeat_times * strings_count).
  __device__ void operator()(size_type const idx) const noexcept
  {
    auto const str_idx    = idx / repeat_times;  // value cycles in [0, string_count)
    auto const repeat_idx = idx % repeat_times;  // value cycles in [0, repeat_times)
    auto const is_valid   = !has_nulls || strings_dv.is_valid_nocheck(str_idx);

    if (!d_chars && repeat_idx == 0) {
      d_offsets[str_idx] =
        is_valid ? repeat_times * strings_dv.element<string_view>(str_idx).size_bytes() : 0;
    }

    // Each input string will be copied by `repeat_times` threads into the output string.
    if (d_chars && is_valid) {
      auto const d_str    = strings_dv.element<string_view>(str_idx);
      auto const str_size = d_str.size_bytes();
      if (str_size > 0) {
        auto const input_ptr  = d_str.data();
        auto const output_ptr = d_chars + d_offsets[str_idx] + repeat_idx * str_size;
        std::memcpy(output_ptr, input_ptr, str_size);
      }
    }
  }
};

/**
 * @brief Check if the size of the output strings column exceeds the maximum indexable value.
 */
void check_overflow(strings_column_view const& input,
                    size_type repeat_times,
                    rmm::cuda_stream_view stream)
{
  size_type size_start, size_end;
  CUDA_TRY(cudaMemcpyAsync(&size_start,
                           input.offsets().template data<size_type>() + input.offset(),
                           sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  CUDA_TRY(
    cudaMemcpyAsync(&size_end,
                    input.offsets().template data<size_type>() + input.offset() + input.size(),
                    sizeof(size_type),
                    cudaMemcpyDeviceToHost,
                    stream.value()));
  stream.synchronize();

  CUDF_EXPECTS(size_end - size_start <= std::numeric_limits<size_type>::max() / repeat_times,
               "The output strings have total size that exceeds the maximum allowed size.");
}

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       check_output_overflow overflow_checking,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(data_type{type_id::STRING}); }

  if (repeat_times <= 0) {
    // If the number of repetitions is not positive, each row of the output strings column will be
    // either an empty string (if the input row is not null), or a null (if the input row is null).
    return generate_empty_output(input, strings_count, stream, mr);
  }

  // If `repeat_times == 1`, just make a copy of the input.
  if (repeat_times == 1) { return std::make_unique<column>(input.parent(), stream, mr); }

  // Check for overflow (whether the total size of the output strings column exceeds
  // numeric_limits<size_type>::max().
  if (overflow_checking == check_output_overflow::YES) {
    check_overflow(input, repeat_times, stream);
  }

  auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
  auto const fn = compute_size_and_repeat_fn{*strings_dv_ptr, repeat_times, input.has_nulls()};
  auto [offsets_column, chars_column] =
    make_strings_children(fn, strings_count * repeat_times, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
}

namespace {
constexpr auto invalid_size = std::numeric_limits<size_type>::lowest();
constexpr auto max_offset   = static_cast<int64_t>(std::numeric_limits<size_type>::max());

template <typename Func>
std::tuple<std::unique_ptr<column>, size_type, rmm::device_buffer, size_type>
generate_offsets_and_null_mask(Func& fn,
                               size_type strings_count,
                               bool generate_null_mask,
                               check_output_overflow overflow_checking,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets    = offsets_view.template data<int32_t>();
  fn.d_offsets      = d_offsets;

  auto validities = rmm::device_uvector<int8_t>(generate_null_mask ? strings_count : 0, stream);
  fn.d_validities = generate_null_mask ? validities.begin() : nullptr;

  // Compute sizes of the output strings (and null mask, if needed).
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator<size_type>(0), strings_count, fn);

  // Compute the offsets values.
  if (overflow_checking == check_output_overflow::YES) {
    CUDA_TRY(cudaMemsetAsync(d_offsets + strings_count, 0, sizeof(offset_type), stream.value()));
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           d_offsets,
                           d_offsets + strings_count + 1,
                           d_offsets,
                           size_type{0},
                           [] __device__(auto const lhs, auto const rhs) {
                             if (lhs == invalid_size || rhs == invalid_size) {
                               return invalid_size;
                             }
                             auto const sum = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs);
                             return sum > max_offset ? invalid_size : static_cast<size_type>(sum);
                           });
  } else {
    thrust::exclusive_scan(
      rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);
  }

  auto const size_bytes = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
  CUDF_EXPECTS(size_bytes != invalid_size,
               "The output strings have total size that exceeds the maximum allowed size.");

  auto [null_mask, null_count] = cudf::detail::valid_if(
    validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);

  return std::make_tuple(std::move(offsets_column),
                         size_bytes,
                         null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                         null_count);
}

template <typename Func>
auto generate_output_strings(Func& fn,
                             size_type strings_count,
                             size_type size_bytes,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  auto chars_column = create_chars_child_column(0, size_bytes, stream, mr);

  // Note that if the output chars column has zero size, the function fn should not be called to
  // avoid accidentally overwriting the offsets.
  if (size_bytes > 0) {
    fn.d_chars = chars_column->mutable_view().template data<char>();
    thrust::for_each_n(
      rmm::exec_policy(stream), thrust::make_counting_iterator<size_type>(0), strings_count, fn);
  }

  return chars_column;
}

/**
 * @brief Functor to compute string sizes and repeat the input strings, each string is repeated by a
 * separate number of times.
 */
template <class IntType>
struct compute_size_and_repeat_separately_fn {
  column_device_view const strings_dv;
  column_device_view const repeat_times_dv;
  bool const strings_has_nulls;
  bool const rtimes_has_nulls;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only repeat strings.
  char* d_chars{nullptr};

  // We may need to set `1` or `0` for the validities of the output strings,
  // and only do that when both input columns have nulls.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const idx) const noexcept
  {
    auto const string_is_valid = !strings_has_nulls || strings_dv.is_valid_nocheck(idx);
    auto const rtimes_is_valid = !rtimes_has_nulls || repeat_times_dv.is_valid_nocheck(idx);

    // Any null input (either string or repeat_times value) will result in a null output.
    auto const is_valid = string_is_valid && rtimes_is_valid;

    // When the input string is null, `repeat_times` is also set to 0.
    // This makes sure that if `repeat_times > 0` then we will always have a valid input string,
    // and if `repeat_times <= 0` we will never copy anything to the output.
    auto const repeat_times = is_valid ? repeat_times_dv.element<IntType>(idx) : IntType{0};

    if (!d_chars) {
      d_offsets[idx] = [&] {
        if (repeat_times <= 0) { return size_type{0}; }
        auto const size = static_cast<int64_t>(strings_dv.element<string_view>(idx).size_bytes()) *
                          static_cast<int64_t>(repeat_times);
        return size > max_offset ? invalid_size : static_cast<size_type>(size);
      }();

      // We will allocate memory for `d_validities` only when both input columns have nulls.
      if (d_validities) { d_validities[idx] = is_valid; }
    }

    if (d_chars && repeat_times > 0) {
      auto const d_str    = strings_dv.element<string_view>(idx);
      auto const str_size = d_str.size_bytes();
      if (str_size > 0) {
        auto const input_ptr = d_str.data();
        auto output_ptr      = d_chars + d_offsets[idx];
        for (IntType repeat_idx = 0; repeat_idx < repeat_times; ++repeat_idx) {
          output_ptr = copy_and_increment(output_ptr, input_ptr, str_size);
        }
      }
    }
  }
};

/**
 * @brief The dispatch functions for repeating strings with separate repeating times.
 *
 * The functions expect that the input `repeat_times` column has a non-bool integer data type (i.e.,
 * it has `cudf::is_index_type` data type).
 */
struct dispatch_repeat_strings_separately_fn {
  template <class T, std::enable_if_t<cudf::is_index_type<T>()>* = nullptr>
  std::tuple<std::unique_ptr<column>, std::unique_ptr<column>, rmm::device_buffer, size_type>
  operator()(size_type strings_count,
             strings_column_view const& input,
             column_view const& repeat_times,
             check_output_overflow overflow_checking,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    auto const strings_dv_ptr      = column_device_view::create(input.parent(), stream);
    auto const repeat_times_dv_ptr = column_device_view::create(repeat_times, stream);
    auto const strings_has_nulls   = input.has_nulls();
    auto const rtimes_has_nulls    = repeat_times.has_nulls();
    auto const generate_new_mask   = strings_has_nulls && rtimes_has_nulls;
    auto fn                        = compute_size_and_repeat_separately_fn<T>{
      *strings_dv_ptr, *repeat_times_dv_ptr, strings_has_nulls, rtimes_has_nulls};

    // If `generate_new_mask == false`, output null_mask and null_count are empty/zero.
    auto [offsets_column, size_bytes, null_mask, null_count] = generate_offsets_and_null_mask(
      fn, strings_count, generate_new_mask, overflow_checking, stream, mr);

    // If only one input column has nulls, we just copy its null mask and null count.
    if (strings_has_nulls ^ rtimes_has_nulls) {
      auto const& col = strings_has_nulls ? input.parent() : repeat_times;
      null_mask       = cudf::detail::copy_bitmask(col, stream, mr);
      null_count      = col.null_count();
    }

    // Now jus generate the output strings.
    auto chars_column = generate_output_strings(fn, strings_count, size_bytes, stream, mr);

    return std::make_tuple(
      std::move(offsets_column), std::move(chars_column), std::move(null_mask), null_count);
  }

  template <class T, std::enable_if_t<!cudf::is_index_type<T>()>* = nullptr>
  std::tuple<std::unique_ptr<column>, std::unique_ptr<column>, rmm::device_buffer, size_type>
  operator()(size_type,
             strings_column_view const&,
             column_view const&,
             check_output_overflow,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("repeat_strings expects an integer type for the `repeat_times` input column.");
  }
};

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       check_output_overflow overflow_checking,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == repeat_times.size(), "The input columns must have the same size.");

  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(data_type{type_id::STRING}); }

  auto [offsets_column, chars_column, null_mask, null_count] =
    type_dispatcher(repeat_times.type(),
                    dispatch_repeat_strings_separately_fn{},
                    strings_count,
                    input,
                    repeat_times,
                    overflow_checking,
                    stream,
                    mr);
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

std::unique_ptr<string_scalar> repeat_string(string_scalar const& input,
                                             size_type repeat_times,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_string(input, repeat_times, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       check_output_overflow overflow_checking,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(
    input, repeat_times, overflow_checking, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       check_output_overflow overflow_checking,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(
    input, repeat_times, overflow_checking, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
