/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "text/normalize.cuh"
#include "text/subword/detail/data_normalizer.hpp"
#include "text/subword/detail/tokenizer_utils.cuh"
#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/normalize.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Normalize spaces in a strings column.
 *
 * Repeated whitespace (code-point <= ' ') is replaced with a single space.
 * Also, whitespace is trimmed from the beginning and end of each string.
 *
 * This functor can be called to compute the output size in bytes
 * of each string and then called again to fill in the allocated buffer.
 */
struct normalize_spaces_fn {
  cudf::column_device_view const d_strings;  // strings to normalize
  cudf::size_type* d_sizes{};                // size of each output row
  char* d_chars{};                           // output buffer for characters
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    cudf::string_view const single_space(" ", 1);
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    char* buffer     = d_chars ? d_chars + d_offsets[idx] : nullptr;
    char* optr       = buffer;  // running output pointer

    cudf::size_type nbytes = 0;  // holds the number of bytes per output string

    // create a tokenizer for this string with whitespace delimiter (default)
    characters_tokenizer tokenizer(d_str);

    // this will retrieve tokens automatically skipping runs of whitespace
    while (tokenizer.next_token()) {
      auto const token_pos = tokenizer.token_byte_positions();
      auto const token =
        cudf::string_view(d_str.data() + token_pos.first, token_pos.second - token_pos.first);
      if (optr) {
        // prepend space unless we are at the beginning
        if (optr != buffer) { optr = cudf::strings::detail::copy_string(optr, single_space); }
        // write token to output buffer
        thrust::copy_n(thrust::seq, token.data(), token.size_bytes(), optr);
        optr += token.size_bytes();
      }
      nbytes += token.size_bytes() + 1;  // token size plus a single space
    }
    // remove trailing space
    if (!d_chars) { d_sizes[idx] = (nbytes > 0) ? nbytes - 1 : 0; }
  }
};

// code-point to multi-byte range limits
constexpr uint32_t UTF8_1BYTE = 0x0080;
constexpr uint32_t UTF8_2BYTE = 0x0800;
constexpr uint32_t UTF8_3BYTE = 0x01'0000;

__device__ int8_t cp_to_utf8(uint32_t codepoint, char* out)
{
  auto utf8 = cudf::strings::detail::codepoint_to_utf8(codepoint);
  return cudf::strings::detail::from_char_utf8(utf8, out);
#if 0
  auto out_ptr = out;
  if (codepoint < UTF8_1BYTE)  // ASCII range
    *out_ptr++ = static_cast<char>(codepoint);
  else if (codepoint < UTF8_2BYTE) {  // create two-byte UTF-8
    // b00001xxx:byyyyyyyy => b110xxxyy:b10yyyyyy
    *out_ptr++ = static_cast<char>((((codepoint << 2) & 0x00'1F00) | 0x00'C000) >> 8);
    *out_ptr++ = static_cast<char>((codepoint & 0x3F) | 0x0080);
  } else if (codepoint < UTF8_3BYTE) {  // create three-byte UTF-8
    // bxxxxxxxx:byyyyyyyy => b1110xxxx:b10xxxxyy:b10yyyyyy
    *out_ptr++ = static_cast<char>((((codepoint << 4) & 0x0F'0000) | 0x00E0'0000) >> 16);
    *out_ptr++ = static_cast<char>((((codepoint << 2) & 0x00'3F00) | 0x00'8000) >> 8);
    *out_ptr++ = static_cast<char>((codepoint & 0x3F) | 0x0080);
  } else {  // create four-byte UTF-8
    // maximum code-point value is 0x0011'0000
    // b000xxxxx:byyyyyyyy:bzzzzzzzz => b11110xxx:b10xxyyyy:b10yyyyzz:b10zzzzzz
    *out_ptr++ = static_cast<char>((((codepoint << 6) & 0x0700'0000u) | 0xF000'0000u) >> 24);
    *out_ptr++ = static_cast<char>((((codepoint << 4) & 0x003F'0000u) | 0x0080'0000u) >> 16);
    *out_ptr++ = static_cast<char>((((codepoint << 2) & 0x00'3F00u) | 0x00'8000u) >> 8);
    *out_ptr++ = static_cast<char>((codepoint & 0x3F) | 0x0080);
  }
  return static_cast<int8_t>(thrust::distance(out, out_ptr));
#endif
}

/**
 * @brief Convert code-point arrays into UTF-8 bytes for each string.
 */
struct codepoint_to_utf8_fn {
  cudf::column_device_view const d_strings;  // input strings
  uint32_t const* cp_data;                   // full code-point array
  int64_t const* d_cp_offsets{};             // offsets to each string's code-point array
  cudf::size_type* d_sizes{};                // size of output string
  char* d_chars{};                           // buffer for the output strings column
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Return the number of bytes for the output string given its code-point array.
   *
   * @param str_cps code-points for the string
   * @param count number of code-points in `str_cps`
   * @return Number of bytes required for the output
   */
  __device__ cudf::size_type compute_output_size(uint32_t const* str_cps, uint32_t count)
  {
    return thrust::transform_reduce(
      thrust::seq,
      str_cps,
      str_cps + count,
      [](auto cp) { return 1 + (cp >= UTF8_1BYTE) + (cp >= UTF8_2BYTE) + (cp >= UTF8_3BYTE); },
      0,
      thrust::plus());
  }

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const offset = d_cp_offsets[idx];
    auto const count  = d_cp_offsets[idx + 1] - offset;  // number of code-points
    auto str_cps      = cp_data + offset;                // code-points for this string
    if (!d_chars) {
      d_sizes[idx] = compute_output_size(str_cps, count);
      return;
    }
    // convert each code-point to 1-4 UTF-8 encoded bytes
    char* out_ptr = d_chars + d_offsets[idx];
    for (uint32_t jdx = 0; jdx < count; ++jdx) {
      uint32_t codepoint = *str_cps++;
      out_ptr += cp_to_utf8(codepoint, out_ptr);
    }
  }
};

}  // namespace

// detail API
std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create device column
  auto d_strings = cudf::column_device_view::create(strings.parent(), stream);

  // build offsets and children using the normalize_space_fn
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    normalize_spaces_fn{*d_strings}, strings.size(), stream, mr);

  return cudf::make_strings_column(strings.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create the normalizer and call it
  auto result = [&] {
    auto const cp_metadata = get_codepoint_metadata(stream);
    auto const aux_table   = get_aux_codepoint_data(stream);
    auto const normalizer  = data_normalizer(cp_metadata.data(), aux_table.data(), do_lower_case);
    return normalizer.normalize(strings, stream);
  }();

  CUDF_EXPECTS(
    result.first->size() < static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "output exceeds the column size limit",
    std::overflow_error);

  // convert the result into a strings column
  // - the cp_chars are the new 4-byte code-point values for all the characters in the output
  // - the cp_offsets identify which code-points go with which strings
  auto const cp_chars   = result.first->data();
  auto const cp_offsets = result.second->data();

  auto d_strings = cudf::column_device_view::create(strings.parent(), stream);

  // build offsets and children using the codepoint_to_utf8_fn
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    codepoint_to_utf8_fn{*d_strings, cp_chars, cp_offsets}, strings.size(), stream, mr);

  return cudf::make_strings_column(strings.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(input, stream, mr);
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   bool do_lower_case,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(input, do_lower_case, stream, mr);
}

struct character_normalizer::character_normalizer_impl {
  rmm::device_uvector<uint32_t> cp_metadata;
  rmm::device_uvector<aux_codepoint_data_type> aux_table;
  bool do_lower_case;
  bool special_tokens;

  character_normalizer_impl(rmm::device_uvector<uint32_t>&& cp_metadata,
                            rmm::device_uvector<aux_codepoint_data_type>&& aux_table,
                            bool do_lower_case,
                            bool special_tokens)
    : cp_metadata(std::move(cp_metadata)),
      aux_table(std::move(aux_table)),
      do_lower_case{do_lower_case},
      special_tokens{special_tokens}
  {
  }
};

character_normalizer::character_normalizer(bool do_lower_case,
                                           bool special_tokens,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref)
{
  auto cp_metadata = nvtext::detail::get_codepoint_metadata(stream);
  auto aux_table   = nvtext::detail::get_aux_codepoint_data(stream);

  _impl = new character_normalizer_impl(
    std::move(cp_metadata), std::move(aux_table), do_lower_case, special_tokens);
}
character_normalizer::~character_normalizer() { delete _impl; }

std::unique_ptr<character_normalizer> create_character_normalizer(bool do_lower_case,
                                                                  bool allow_special_tokens,
                                                                  rmm::cuda_stream_view stream,
                                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<character_normalizer>(do_lower_case, allow_special_tokens, stream, mr);
}

namespace detail {
namespace {
CUDF_KERNEL void normalizer_kernel(char const* d_chars,
                                   int64_t total_bytes,
                                   codepoint_metadata_type const* cp_metadata,
                                   aux_codepoint_data_type const* aux_table,
                                   bool do_lower_case,
                                   bool,  // allow_special_tokens,
                                   uint32_t* d_output,
                                   int8_t* chars_per_thread)
{
  uint32_t replacement[MAX_NEW_CHARS] = {0};

  auto const idx       = cudf::detail::grid_1d::global_thread_id();
  int8_t num_new_chars = 0;

  if ((idx < total_bytes) && cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) {
    auto const cp = [utf8 = d_chars + idx] {
      cudf::char_utf8 ch_utf8;
      auto const ch_size = cudf::strings::detail::to_char_utf8(utf8, ch_utf8);
      return cudf::strings::detail::utf8_to_codepoint(ch_utf8);
    }();
    auto const metadata = cp_metadata[cp];

    if (!should_remove_cp(metadata, do_lower_case)) {
      num_new_chars = 1;
      // Apply lower cases and accent stripping if necessary
      auto const new_cp = do_lower_case || always_replace(metadata) ? get_first_cp(metadata) : cp;
      replacement[0]    = new_cp == 0 ? cp : new_cp;

      if (do_lower_case && is_multi_char_transform(metadata)) {
        auto const next_cps = aux_table[cp];
        replacement[1]      = static_cast<uint32_t>(next_cps >> 32);
        replacement[2]      = static_cast<uint32_t>(next_cps & 0xFFFFFFFF);
        num_new_chars       = 2 + (replacement[2] != 0);
      }

      // check for possible special tokens here before checking add-spaces?

      if (should_add_spaces(metadata, do_lower_case) && (num_new_chars == 1)) {
        // Need to shift all existing code-points up one.
        // This is a rotate right. There is no thrust equivalent at this time.
        // for (int loc = num_new_chars; loc > 0; --loc) {
        //  replacement[loc] = replacement[loc - 1];
        //}
        // Write the required spaces at each end
        replacement[1] = replacement[0];
        replacement[0] = SPACE_CODE_POINT;
        replacement[2] = SPACE_CODE_POINT;
        num_new_chars  = 3;
      }

      // convert back to UTF-8
      for (int k = 0; k < num_new_chars; ++k) {
        auto const new_cp = replacement[k];
        if (new_cp) { cp_to_utf8(new_cp, reinterpret_cast<char*>(replacement + k)); }
      }
    }
  }

  if (idx < total_bytes) { chars_per_thread[idx] = num_new_chars; }

  using BlockStore =
    cub::BlockStore<uint32_t, THREADS_PER_BLOCK, MAX_NEW_CHARS, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ typename BlockStore::TempStorage temp_storage;

  // Now we perform coalesced writes back to global memory using cub.
  auto output_offset = blockIdx.x * blockDim.x * MAX_NEW_CHARS;
  auto block_base    = d_output + output_offset;
  auto valid_items =
    min(static_cast<int>(total_bytes - output_offset), static_cast<int>(blockDim.x));
  BlockStore(temp_storage).Store(block_base, replacement, valid_items);
}

template <typename OffsetType>
rmm::device_uvector<cudf::size_type> compute_sizes(int8_t const* sizes,
                                                   OffsetType offsets,
                                                   int64_t offset,
                                                   cudf::size_type size,
                                                   rmm::cuda_stream_view stream)
{
  auto output_sizes = rmm::device_uvector<cudf::size_type>(size, stream);

  auto d_in        = sizes;
  auto d_out       = output_sizes.begin();
  std::size_t temp = 0;
  nvtxRangePushA("segmented_reduce");
  if (offset == 0) {
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets, offsets + 1, stream.value());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets, offsets + 1, stream.value());
  } else {
    // offsets need to be normalized for segmented-reduce to work efficiently
    auto d_offsets = rmm::device_uvector<cudf::size_type>(size + 1, stream);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      offsets,
                      offsets + size + 1,
                      d_offsets.begin(),
                      [offset] __device__(auto o) { return o - offset; });
    auto const offsets_itr = d_offsets.begin();
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.value());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.value());
  }
  stream.synchronize();
  nvtxRangePop();

  return output_sizes;
}
}  // namespace
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars_size(stream)
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  constexpr int64_t block_size = 64;
  cudf::detail::grid_1d grid{chars_size, block_size};
  auto const max_new_char_total = MAX_NEW_CHARS * chars_size;

  auto d_code_points = rmm::device_uvector<uint32_t>(max_new_char_total, stream);
  auto d_sizes       = rmm::device_uvector<int8_t>(chars_size, stream);
  normalizer_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    d_input_chars,
    chars_size,
    normalizer._impl->cp_metadata.data(),
    normalizer._impl->aux_table.data(),
    normalizer._impl->do_lower_case,
    normalizer._impl->special_tokens,
    d_code_points.data(),
    d_sizes.data());

  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // use segmented-reduce with input_offsets over d_sizes to get the size of the output rows
  auto output_sizes =
    compute_sizes(d_sizes.data(), input_offsets, first_offset, input.size(), stream);

  // convert the sizes to offsets
  auto [offsets, total_size] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  // create output chars and use remove-copy(0) on d_code_points
  rmm::device_uvector<char> chars(total_size, stream, mr);
  auto begin = reinterpret_cast<char const*>(d_code_points.begin());
  auto end   = reinterpret_cast<char const*>(d_code_points.end());
  thrust::remove_copy(rmm::exec_policy_nosync(stream), begin, end, chars.data(), 0);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(input, normalizer, stream, mr);
}

}  // namespace nvtext
