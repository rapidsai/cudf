/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

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
 * @brief Generate a strings column in which each row is an empty string or a null.
 *
 * The output strings column has the same bitmask as the input column.
 */
auto generate_empty_output(strings_column_view const& input,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  auto chars_column = create_chars_child_column(0, stream, mr);

  auto offsets_column = make_numeric_column(
    data_type{type_to_id<offset_type>()}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(offsets_column->mutable_view().template data<offset_type>(),
                                0,
                                offsets_column->size() * sizeof(offset_type),
                                stream.value()));

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

/**
 * @brief Functor to compute output string sizes and repeat the input strings.
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

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  if (repeat_times <= 0) {
    // If the number of repetitions is not positive, each row of the output strings column will be
    // either an empty string (if the input row is not null), or a null (if the input row is null).
    return generate_empty_output(input, strings_count, stream, mr);
  }

  // If `repeat_times == 1`, just make a copy of the input.
  if (repeat_times == 1) { return std::make_unique<column>(input.parent(), stream, mr); }

  auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
  auto const fn = compute_size_and_repeat_fn{*strings_dv_ptr, repeat_times, input.has_nulls()};

  auto [offsets_column, chars_column] =
    make_strings_children(fn, strings_count * repeat_times, strings_count, stream, mr);
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

namespace {
/**
 * @brief Functor to compute string sizes and repeat the input strings, each string is repeated by a
 * separate number of times.
 */
template <class Iterator>
struct compute_size_and_repeat_separately_fn {
  column_device_view const strings_dv;
  column_device_view const repeat_times_dv;
  Iterator const repeat_times_iter;
  bool const strings_has_nulls;
  bool const rtimes_has_nulls;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only repeat strings.
  char* d_chars{nullptr};

  __device__ int64_t operator()(size_type const idx) const noexcept
  {
    auto const string_is_valid = !strings_has_nulls || strings_dv.is_valid_nocheck(idx);
    auto const rtimes_is_valid = !rtimes_has_nulls || repeat_times_dv.is_valid_nocheck(idx);

    // Any null input (either string or repeat_times value) will result in a null output.
    auto const is_valid = string_is_valid && rtimes_is_valid;

    // When the input string is null, `repeat_times` and `string_size` are also set to 0.
    // This makes sure that if `repeat_times > 0` then we will always have a valid input string,
    // and if `repeat_times <= 0` we will never copy anything to the output.
    auto const repeat_times = is_valid ? repeat_times_iter[idx] : size_type{0};
    auto const string_size =
      is_valid ? strings_dv.element<string_view>(idx).size_bytes() : size_type{0};

    // The output_size is returned, and it needs to be an int64_t number to prevent overflow.
    auto const output_size =
      repeat_times > 0 ? static_cast<int64_t>(repeat_times) * static_cast<int64_t>(string_size)
                       : int64_t{0};

    if (!d_chars) {
      // If overflow happen, the stored value of output string size will be incorrect due to
      // downcasting. In such cases, the entire output string size array should be discarded.
      d_offsets[idx] = static_cast<offset_type>(output_size);
    } else if (repeat_times > 0 && string_size > 0) {
      auto const d_str     = strings_dv.element<string_view>(idx);
      auto const input_ptr = d_str.data();
      auto output_ptr      = d_chars + d_offsets[idx];
      for (size_type repeat_idx = 0; repeat_idx < repeat_times; ++repeat_idx) {
        output_ptr = copy_and_increment(output_ptr, input_ptr, string_size);
      }
    }

    // The output_size value may be used to sum up to detect overflow at the caller site.
    // The caller can detect overflow easily by checking `SUM(output_size) > INT_MAX`.
    return output_size;
  }
};

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * This function is similar to `strings::detail::make_strings_children`, except that it accepts an
 * optional input `std::optional<column_view>` that can contain the precomputed sizes of the output
 * strings.
 */
template <typename Func>
auto make_strings_children(Func fn,
                           size_type exec_size,
                           size_type strings_count,
                           std::optional<column_view> output_strings_sizes,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);

  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets    = offsets_view.template data<size_type>();
  fn.d_offsets      = d_offsets;

  // This may be called twice -- once for offsets and once for chars.
  auto for_each_fn = [exec_size, stream](Func& fn) {
    thrust::for_each_n(
      rmm::exec_policy(stream), thrust::make_counting_iterator<size_type>(0), exec_size, fn);
  };

  if (!output_strings_sizes.has_value()) {
    // Compute the output sizes only if they are not given.
    for_each_fn(fn);

    // Compute the offsets values.
    thrust::exclusive_scan(
      rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);
  } else {
    // Compute the offsets values from the provided output string sizes.
    auto const string_sizes = output_strings_sizes.value();
    CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(offset_type), stream.value()));
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           string_sizes.template begin<size_type>(),
                           string_sizes.template end<size_type>(),
                           d_offsets + 1);
  }

  // Now build the chars column
  auto const bytes  = cudf::detail::get_value<size_type>(offsets_view, strings_count, stream);
  auto chars_column = create_chars_child_column(bytes, stream, mr);

  // Execute the function fn again to fill the chars column.
  // Note that if the output chars column has zero size, the function fn should not be called to
  // avoid accidentally overwriting the offsets.
  if (bytes > 0) {
    fn.d_chars = chars_column->mutable_view().template data<char>();
    for_each_fn(fn);
  }

  return std::pair(std::move(offsets_column), std::move(chars_column));
}

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       std::optional<column_view> output_strings_sizes,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == repeat_times.size(), "The input columns must have the same size.");
  CUDF_EXPECTS(cudf::is_index_type(repeat_times.type()),
               "repeat_strings expects an integer type for the `repeat_times` input column.");
  if (output_strings_sizes.has_value()) {
    auto const output_sizes = output_strings_sizes.value();
    CUDF_EXPECTS(input.size() == output_sizes.size() &&
                   (!output_sizes.nullable() || !output_sizes.has_nulls()),
                 "The given column of output string sizes is invalid.");
  }

  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  auto const strings_dv_ptr      = column_device_view::create(input.parent(), stream);
  auto const repeat_times_dv_ptr = column_device_view::create(repeat_times, stream);
  auto const strings_has_nulls   = input.has_nulls();
  auto const rtimes_has_nulls    = repeat_times.has_nulls();
  auto const repeat_times_iter =
    cudf::detail::indexalator_factory::make_input_iterator(repeat_times);
  auto const fn = compute_size_and_repeat_separately_fn<decltype(repeat_times_iter)>{
    *strings_dv_ptr, *repeat_times_dv_ptr, repeat_times_iter, strings_has_nulls, rtimes_has_nulls};

  auto [offsets_column, chars_column] =
    make_strings_children(fn, strings_count, strings_count, output_strings_sizes, stream, mr);

  // We generate new bitmask by AND of the input columns' bitmasks.
  // Note that if the input columns are nullable, the output column will also be nullable (which may
  // not have nulls).
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{input.parent(), repeat_times}}, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask));
}

std::pair<std::unique_ptr<column>, int64_t> repeat_strings_output_sizes(
  strings_column_view const& input,
  column_view const& repeat_times,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == repeat_times.size(), "The input columns must have the same size.");
  CUDF_EXPECTS(
    cudf::is_index_type(repeat_times.type()),
    "repeat_strings_output_sizes expects an integer type for the `repeat_times` input column.");

  auto const strings_count = input.size();
  if (strings_count == 0) {
    return std::pair(make_empty_column(type_to_id<size_type>()), int64_t{0});
  }

  auto output_sizes = make_numeric_column(
    data_type{type_to_id<size_type>()}, strings_count, mask_state::UNALLOCATED, stream, mr);

  auto const strings_dv_ptr      = column_device_view::create(input.parent(), stream);
  auto const repeat_times_dv_ptr = column_device_view::create(repeat_times, stream);
  auto const strings_has_nulls   = input.has_nulls();
  auto const rtimes_has_nulls    = repeat_times.has_nulls();
  auto const repeat_times_iter =
    cudf::detail::indexalator_factory::make_input_iterator(repeat_times);

  auto const fn = compute_size_and_repeat_separately_fn<decltype(repeat_times_iter)>{
    *strings_dv_ptr,
    *repeat_times_dv_ptr,
    repeat_times_iter,
    strings_has_nulls,
    rtimes_has_nulls,
    output_sizes->mutable_view().template begin<size_type>()};

  auto const total_bytes =
    thrust::transform_reduce(rmm::exec_policy(stream),
                             thrust::make_counting_iterator<size_type>(0),
                             thrust::make_counting_iterator<size_type>(strings_count),
                             fn,
                             int64_t{0},
                             thrust::plus{});

  return std::pair(std::move(output_sizes), total_bytes);
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
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       column_view const& repeat_times,
                                       std::optional<column_view> output_strings_sizes,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(
    input, repeat_times, output_strings_sizes, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<column>, int64_t> repeat_strings_output_sizes(
  strings_column_view const& input,
  column_view const& repeat_times,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings_output_sizes(input, repeat_times, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
