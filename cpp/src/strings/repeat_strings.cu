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
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<string_scalar> repeat_strings(string_scalar const& input,
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
  auto chars_column = create_chars_child_column(0, stream, mr);

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

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
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

  auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
  auto const fn = compute_size_and_repeat_fn{*strings_dv_ptr, repeat_times, input.has_nulls()};

  // Repeat the strings in each row.
  // Note that this cannot handle the cases when the size of the output column exceeds the maximum
  // value that can be indexed by size_type (offset_type).
  // In such situations, an exception may be thrown, or the output result is undefined.
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

}  // namespace detail

std::unique_ptr<string_scalar> repeat_strings(string_scalar const& input,
                                              size_type repeat_times,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
