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
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvtext/detail/generate_ngrams.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

#include <stdexcept>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Generate ngrams from strings column.
 *
 * Adjacent strings are concatenated with the provided separator.
 * The number of adjacent strings join depends on the specified ngrams value.
 * For example: for bigrams (ngrams=2), pairs of strings are concatenated.
 */
struct ngram_generator_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type ngrams;
  cudf::string_view const d_separator;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Build ngram for each string.
   *
   * This is called for each thread and processed for each string.
   * Each string will produce the number of ngrams specified.
   *
   * @param idx Index of the kernel thread.
   * @return Number of bytes required for the string for this thread.
   */
  __device__ void operator()(cudf::size_type idx)
  {
    char* out_ptr         = d_chars ? d_chars + d_offsets[idx] : nullptr;
    cudf::size_type bytes = 0;
    for (cudf::size_type n = 0; n < ngrams; ++n) {
      auto const d_str = d_strings.element<cudf::string_view>(n + idx);
      bytes += d_str.size_bytes();
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_str);
      if ((n + 1) >= ngrams) continue;
      bytes += d_separator.size_bytes();
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator);
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

std::unique_ptr<cudf::column> generate_ngrams(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    separator.is_valid(stream), "Parameter separator must be valid", std::invalid_argument);
  cudf::string_view const d_separator(separator.data(), separator.size());
  CUDF_EXPECTS(ngrams > 1,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  auto strings_count = strings.size();
  if (strings_count == 0)  // if no strings, return an empty column
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // first create a new offsets vector removing nulls and empty strings from the input column
  std::unique_ptr<cudf::column> non_empty_offsets_column = [&] {
    cudf::column_view offsets_view(
      strings.offsets().type(), strings_count + 1, strings.offsets().head(), nullptr, 0);
    auto table_offsets = cudf::detail::copy_if(
                           cudf::table_view({offsets_view}),
                           [d_strings, strings_count] __device__(cudf::size_type idx) {
                             if (idx == strings_count) return true;
                             if (d_strings.is_null(idx)) return false;
                             return !d_strings.element<cudf::string_view>(idx).empty();
                           },
                           stream,
                           rmm::mr::get_current_device_resource())
                           ->release();
    strings_count = table_offsets.front()->size() - 1;
    auto result   = std::move(table_offsets.front());
    return result;
  }();  // this allows freeing the temporary table_offsets

  CUDF_EXPECTS(strings_count >= ngrams, "Insufficient number of strings to generate ngrams");
  // create a temporary column view from the non-empty offsets and chars column views
  cudf::column_view strings_view(cudf::data_type{cudf::type_id::STRING},
                                 strings_count,
                                 strings.chars_begin(stream),
                                 nullptr,
                                 0,
                                 0,
                                 {non_empty_offsets_column->view()});
  strings_column = cudf::column_device_view::create(strings_view, stream);
  d_strings      = *strings_column;

  // compute the number of strings of ngrams
  auto const ngrams_count = strings_count - ngrams + 1;

  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    ngram_generator_fn{d_strings, ngrams, d_separator}, ngrams_count, stream, mr);

  // make the output strings column from the offsets and chars column
  return cudf::make_strings_column(
    ngrams_count, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<cudf::column> generate_ngrams(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::generate_ngrams(strings, ngrams, separator, stream, mr);
}

namespace detail {
namespace {

/**
 * @brief Generate character ngrams for each string
 *
 * Each string produces many strings depending on the ngram width and the string size.
 * This functor can be used with `make_strings_children` to build the offsets and
 * the chars child columns.
 */
struct character_ngram_generator_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type ngrams;
  cudf::size_type const* d_ngram_offsets{};
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;
    auto itr                = d_str.begin();
    auto const ngram_offset = d_ngram_offsets[idx];
    auto const ngram_count  = d_ngram_offsets[idx + 1] - ngram_offset;
    auto d_output_sizes     = d_sizes + ngram_offset;
    auto out_ptr            = d_chars ? d_chars + d_offsets[ngram_offset] : nullptr;
    for (cudf::size_type n = 0; n < ngram_count; ++n, ++itr) {
      auto const begin = itr.byte_offset();
      auto const end   = (itr + ngrams).byte_offset();
      if (d_chars) {
        out_ptr =
          cudf::strings::detail::copy_and_increment(out_ptr, d_str.data() + begin, (end - begin));
      } else {
        *d_output_sizes++ = end - begin;
      }
    }
  }
};
}  // namespace

std::unique_ptr<cudf::column> generate_character_ngrams(cudf::strings_column_view const& input,
                                                        cudf::size_type ngrams,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  auto const strings_count = input.size();
  if (strings_count == 0) {  // if no strings, return an empty column
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [d_strings = *d_strings, ngrams] __device__(auto idx) {
        if (d_strings.is_null(idx)) { return 0; }
        auto const length = d_strings.element<cudf::string_view>(idx).length();
        return std::max(0, static_cast<cudf::size_type>(length + 1 - ngrams));
      }));
  auto [offsets, total_ngrams] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + input.size(), stream, mr);
  auto d_offsets = offsets->view().data<cudf::size_type>();
  CUDF_EXPECTS(total_ngrams > 0,
               "Insufficient number of characters in each string to generate ngrams");

  character_ngram_generator_fn generator{*d_strings, ngrams, d_offsets};
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    generator, strings_count, total_ngrams, stream, mr);

  auto output = cudf::make_strings_column(
    total_ngrams, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});

  return make_lists_column(
    input.size(), std::move(offsets), std::move(output), 0, rmm::device_buffer{}, stream, mr);
}

namespace {
/**
 * @brief Computes the hash of each character ngram
 *
 * Each thread processes a single string. Substrings are resolved for every character
 * of the string and hashed.
 */
struct character_ngram_hash_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type ngrams;
  cudf::size_type const* d_ngram_offsets;
  cudf::hash_value_type* d_results;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;
    auto itr                = d_str.begin();
    auto const ngram_offset = d_ngram_offsets[idx];
    auto const ngram_count  = d_ngram_offsets[idx + 1] - ngram_offset;
    auto const hasher       = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{0};
    auto d_hashes           = d_results + ngram_offset;
    for (cudf::size_type n = 0; n < ngram_count; ++n, ++itr) {
      auto const begin = itr.byte_offset();
      auto const end   = (itr + ngrams).byte_offset();
      auto const ngram = cudf::string_view(d_str.data() + begin, end - begin);
      *d_hashes++      = hasher(ngram);
    }
  }
};
}  // namespace

std::unique_ptr<cudf::column> hash_character_ngrams(cudf::strings_column_view const& input,
                                                    cudf::size_type ngrams,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  auto output_type = cudf::data_type{cudf::type_to_id<cudf::hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  // build offsets column by computing the number of ngrams per string
  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [d_strings = *d_strings, ngrams] __device__(auto idx) {
        if (d_strings.is_null(idx)) { return 0; }
        auto const length = d_strings.element<cudf::string_view>(idx).length();
        return std::max(0, static_cast<cudf::size_type>(length + 1 - ngrams));
      }));
  auto [offsets, total_ngrams] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + input.size(), stream, mr);
  auto d_offsets = offsets->view().data<cudf::size_type>();

  CUDF_EXPECTS(total_ngrams > 0,
               "Insufficient number of characters in each string to generate ngrams");

  // compute ngrams and build hashes
  auto hashes =
    cudf::make_numeric_column(output_type, total_ngrams, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_hashes = hashes->mutable_view().data<cudf::hash_value_type>();

  character_ngram_hash_fn generator{*d_strings, ngrams, d_offsets, d_hashes};
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     input.size(),
                     generator);

  return make_lists_column(
    input.size(), std::move(offsets), std::move(hashes), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> generate_character_ngrams(cudf::strings_column_view const& strings,
                                                        cudf::size_type ngrams,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::generate_character_ngrams(strings, ngrams, stream, mr);
}

std::unique_ptr<cudf::column> hash_character_ngrams(cudf::strings_column_view const& strings,
                                                    cudf::size_type ngrams,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash_character_ngrams(strings, ngrams, stream, mr);
}

}  // namespace nvtext
