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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/transform.h>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {
namespace {

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_type split_align = 64;

__device__ size_type compute_memory_size(size_type token_count, size_type token_size_sum)
{
  return cudf::detail::round_up_pow2(token_size_sum, split_align) +
         cudf::detail::round_up_pow2((token_count + 1) * static_cast<size_type>(sizeof(size_type)),
                                     split_align);
}

struct copy_info {
  size_type idx{};
  size_type token_count{};
  size_type token_size_sum{};
  void* memory_ptr{};
};

enum class Dir { FORWARD, BACKWARD };

/**
 * @brief Compute the number of tokens, the total byte sizes of the tokens, and
 * required memory size for the `idx'th` string element of `d_strings`.
 */
template <Dir dir>
struct token_reader_fn {
  column_device_view const d_strings;  // strings to split
  string_view const d_delimiter;       // delimiter for split
  size_type const max_tokens = std::numeric_limits<size_type>::max();
  bool const has_validity    = false;

  template <bool last>
  __device__ size_type compute_token_char_bytes(string_view const& d_str,
                                                size_type start_pos,
                                                size_type end_pos,
                                                size_type delimiter_pos) const
  {
    if (last) {
      return dir == Dir::FORWARD ? d_str.byte_offset(end_pos) - d_str.byte_offset(start_pos)
                                 : d_str.byte_offset(end_pos);
    } else {
      return dir == Dir::FORWARD ? d_str.byte_offset(delimiter_pos) - d_str.byte_offset(start_pos)
                                 : d_str.byte_offset(end_pos) -
                                     d_str.byte_offset(delimiter_pos + d_delimiter.length());
    }
  }

  // returns a tuple of token count, sum of token sizes in bytes, and required
  // memory block size
  __device__ thrust::tuple<size_type, size_type, size_type> operator()(size_type idx) const
  {
    if (has_validity && d_strings.is_null(idx)) {
      return thrust::make_tuple<size_type, size_type, size_type>(0, 0, 0);
    }

    auto const d_str         = d_strings.element<string_view>(idx);
    size_type token_count    = 0;
    size_type token_size_sum = 0;
    size_type start_pos      = 0;               // updates only if moving forward
    auto end_pos             = d_str.length();  // updates only if moving backward
    while (token_count < max_tokens - 1) {
      auto const delimiter_pos = dir == Dir::FORWARD ? d_str.find(d_delimiter, start_pos)
                                                     : d_str.rfind(d_delimiter, start_pos, end_pos);
      if (delimiter_pos != -1) {
        token_count++;
        token_size_sum += compute_token_char_bytes<false>(d_str, start_pos, end_pos, delimiter_pos);
        if (dir == Dir::FORWARD) {
          start_pos = delimiter_pos + d_delimiter.length();
        } else {
          end_pos = delimiter_pos;
        }
      } else {
        break;
      }
    }
    token_count++;
    token_size_sum += compute_token_char_bytes<true>(d_str, start_pos, end_pos, -1);

    auto const memory_size = compute_memory_size(token_count, token_size_sum);

    return thrust::make_tuple<size_type, size_type, size_type>(
      token_count, token_size_sum, memory_size);
  }
};

/**
 * @brief Copy the tokens from the `idx'th` string element of `d_strings` to
 * the contiguous memory buffer.
 */
template <Dir dir>
struct token_copier_fn {
  column_device_view const d_strings;  // strings to split
  string_view const d_delimiter;       // delimiter for split
  bool const has_validity = false;

  template <bool last>
  __device__ thrust::pair<size_type, size_type> compute_src_byte_offset_and_token_char_bytes(
    string_view const& d_str, size_type start_pos, size_type end_pos, size_type delimiter_pos) const
  {
    if (last) {
      auto const src_byte_offset  = dir == Dir::FORWARD ? d_str.byte_offset(start_pos) : 0;
      auto const token_char_bytes = dir == Dir::FORWARD
                                      ? d_str.byte_offset(end_pos) - src_byte_offset
                                      : d_str.byte_offset(end_pos);
      return thrust::make_pair<size_type, size_type>(src_byte_offset, token_char_bytes);
    } else {
      auto const src_byte_offset = dir == Dir::FORWARD
                                     ? d_str.byte_offset(start_pos)
                                     : d_str.byte_offset(delimiter_pos + d_delimiter.length());
      auto const token_char_bytes = dir == Dir::FORWARD
                                      ? d_str.byte_offset(delimiter_pos) - src_byte_offset
                                      : d_str.byte_offset(end_pos) - src_byte_offset;
      return thrust::make_pair<size_type, size_type>(src_byte_offset, token_char_bytes);
    }
  }

  __device__ void operator()(copy_info const info) const
  {
    if (info.token_count == 0) { return; }

    auto memory_ptr = static_cast<char*>(info.memory_ptr);

    auto const char_buf_size = cudf::detail::round_up_pow2(info.token_size_sum, split_align);
    auto const char_buf_ptr  = memory_ptr;
    memory_ptr += char_buf_size;
    auto const offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

    auto const d_str            = d_strings.element<string_view>(info.idx);
    size_type token_idx         = 0;
    size_type char_bytes_copied = 0;
    size_type start_pos         = 0;               // updates only if moving forward
    auto end_pos                = d_str.length();  // updates only if moving backward
    while (token_idx < info.token_count - 1) {
      auto const delimiter_pos = dir == Dir::FORWARD ? d_str.find(d_delimiter, start_pos)
                                                     : d_str.rfind(d_delimiter, start_pos, end_pos);
      if (delimiter_pos != -1) {
        auto const offset_size_pair = compute_src_byte_offset_and_token_char_bytes<false>(
          d_str, start_pos, end_pos, delimiter_pos);
        if (dir == Dir::FORWARD) {
          thrust::copy(thrust::seq,
                       d_str.data() + offset_size_pair.first,
                       d_str.data() + offset_size_pair.first + offset_size_pair.second,
                       char_buf_ptr + char_bytes_copied);
          offset_buf_ptr[token_idx] = char_bytes_copied;
        } else {
          auto const char_buf_offset =
            info.token_size_sum - char_bytes_copied - offset_size_pair.second;
          thrust::copy(thrust::seq,
                       d_str.data() + offset_size_pair.first,
                       d_str.data() + offset_size_pair.first + offset_size_pair.second,
                       char_buf_ptr + char_buf_offset);
          offset_buf_ptr[info.token_count - 1 - token_idx] = char_buf_offset;
        }
        token_idx++;
        char_bytes_copied += offset_size_pair.second;
        if (dir == Dir::FORWARD) {
          start_pos = delimiter_pos + d_delimiter.length();
        } else {
          end_pos = delimiter_pos;
        }
      } else {
        break;
      }
    }

    auto const offset_size_pair =
      compute_src_byte_offset_and_token_char_bytes<true>(d_str, start_pos, end_pos, -1);
    if (dir == Dir::FORWARD) {
      thrust::copy(thrust::seq,
                   d_str.data() + offset_size_pair.first,
                   d_str.data() + offset_size_pair.first + offset_size_pair.second,
                   char_buf_ptr + char_bytes_copied);
      offset_buf_ptr[token_idx] = char_bytes_copied;
    } else {
      thrust::copy(thrust::seq, d_str.data(), d_str.data() + offset_size_pair.second, char_buf_ptr);
      offset_buf_ptr[0] = 0;
    }
    offset_buf_ptr[info.token_count] = info.token_size_sum;
  }
};

/**
 * @brief Compute the number of tokens, the total byte sizes of the tokens, and
 * required memory size for the `idx'th` string element of `d_strings`.
 */
template <Dir dir>
struct whitespace_token_reader_fn {
  column_device_view const d_strings;  // strings to split
  size_type const max_tokens = std::numeric_limits<size_type>::max();
  bool const has_validity    = false;

  template <bool last>
  __device__ size_type compute_token_char_bytes(string_view const& d_str,
                                                size_type cur_pos,
                                                size_type to_token_pos) const
  {
    if (last) {
      return dir == Dir::FORWARD
               ? d_str.byte_offset(d_str.length()) - d_str.byte_offset(to_token_pos)
               : d_str.byte_offset(to_token_pos + 1) - d_str.byte_offset(0);
    } else {
      return dir == Dir::FORWARD
               ? d_str.byte_offset(cur_pos) - d_str.byte_offset(to_token_pos)
               : d_str.byte_offset(to_token_pos + 1) - d_str.byte_offset(cur_pos + 1);
    }
  }

  __device__ thrust::tuple<size_type, size_type, size_type> operator()(size_type idx) const
  {
    if (has_validity && d_strings.is_null(idx)) {
      return thrust::make_tuple<size_type, size_type, size_type>(0, 0, 0);
    }

    auto const d_str         = d_strings.element<string_view>(idx);
    size_type token_count    = 0;
    size_type token_size_sum = 0;
    auto spaces              = true;
    auto reached_max_tokens  = false;
    size_type to_token_pos   = 0;
    for (size_type i = 0; i < d_str.length(); ++i) {
      auto const cur_pos = dir == Dir::FORWARD ? i : d_str.length() - 1 - i;
      auto const ch      = d_str[cur_pos];
      if (spaces != (ch <= ' ')) {
        if (spaces) {  // from whitespace(s) to a new token
          to_token_pos = cur_pos;
        } else {  // from a token to whitespace(s)
          if (token_count < max_tokens - 1) {
            token_count++;
            token_size_sum += compute_token_char_bytes<false>(d_str, cur_pos, to_token_pos);
          } else {
            reached_max_tokens = true;
            break;
          }
        }
        spaces = !spaces;
      }
    }
    if (reached_max_tokens || !spaces) {
      token_count++;
      token_size_sum += compute_token_char_bytes<true>(d_str, -1, to_token_pos);
    }

    if (token_count == 0) {  // note that pandas.Series.str.split("", pat=" ")
                             // returns one token (i.e. "") while
                             // pandas.Series.str.split("") returns 0 token.
      return thrust::make_tuple<size_type, size_type, size_type>(0, 0, 0);
    }

    auto const memory_size = compute_memory_size(token_count, token_size_sum);

    return thrust::make_tuple<size_type, size_type, size_type>(
      token_count, token_size_sum, memory_size);
  }
};

/**
 * @brief Copy the tokens from the `idx'th` string element of `d_strings` to
 * the contiguous memory buffer.
 */
template <Dir dir>
struct whitespace_token_copier_fn {
  column_device_view const d_strings;  // strings to split
  bool const has_validity = false;

  template <bool last>
  __device__ thrust::pair<size_type, size_type> compute_src_byte_offset_and_token_char_bytes(
    string_view const& d_str,
    size_type cur_pos,
    size_type to_token_pos,
    size_type remaining_bytes) const
  {
    if (last) {
      auto const token_char_bytes = remaining_bytes;
      auto const src_byte_offset  = dir == Dir::FORWARD
                                     ? d_str.byte_offset(to_token_pos)
                                     : d_str.byte_offset(to_token_pos + 1) - token_char_bytes;
      return thrust::make_pair<size_type, size_type>(src_byte_offset, token_char_bytes);
    } else {
      auto const src_byte_offset =
        dir == Dir::FORWARD ? d_str.byte_offset(to_token_pos) : d_str.byte_offset(cur_pos + 1);
      auto const token_char_bytes = dir == Dir::FORWARD
                                      ? d_str.byte_offset(cur_pos) - src_byte_offset
                                      : d_str.byte_offset(to_token_pos + 1) - src_byte_offset;
      return thrust::make_pair<size_type, size_type>(src_byte_offset, token_char_bytes);
    }
  }

  __device__ void operator()(copy_info const info) const
  {
    if (info.token_count == 0) { return; }

    auto memory_ptr = static_cast<char*>(info.memory_ptr);

    auto const char_buf_size = cudf::detail::round_up_pow2(info.token_size_sum, split_align);
    auto const char_buf_ptr  = memory_ptr;
    memory_ptr += char_buf_size;
    auto const offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

    auto const d_str            = d_strings.element<string_view>(info.idx);
    size_type token_idx         = 0;
    size_type char_bytes_copied = 0;
    auto spaces                 = true;
    size_type to_token_pos      = 0;
    for (size_type i = 0; i < d_str.length(); ++i) {
      auto const cur_pos = dir == Dir::FORWARD ? i : d_str.length() - 1 - i;
      auto const ch      = d_str[cur_pos];
      if (spaces != (ch <= ' ')) {
        if (spaces) {  // from whitespace(s) to a new token
          to_token_pos = cur_pos;
        } else {  // from a token to whitespace(s)
          if (token_idx < info.token_count - 1) {
            auto const offset_size_pair = compute_src_byte_offset_and_token_char_bytes<false>(
              d_str, cur_pos, to_token_pos, info.token_size_sum - char_bytes_copied);
            if (dir == Dir::FORWARD) {
              thrust::copy(thrust::seq,
                           d_str.data() + offset_size_pair.first,
                           d_str.data() + offset_size_pair.first + offset_size_pair.second,
                           char_buf_ptr + char_bytes_copied);
              offset_buf_ptr[token_idx] = char_bytes_copied;
            } else {
              auto const char_buf_offset =
                info.token_size_sum - char_bytes_copied - offset_size_pair.second;
              thrust::copy(thrust::seq,
                           d_str.data() + offset_size_pair.first,
                           d_str.data() + offset_size_pair.first + offset_size_pair.second,
                           char_buf_ptr + char_buf_offset);
              offset_buf_ptr[info.token_count - 1 - token_idx] = char_buf_offset;
            }
            token_idx++;
            char_bytes_copied += offset_size_pair.second;
          } else {
            break;
          }
        }
        spaces = !spaces;
      }
    }
    if (token_idx < info.token_count) {
      auto const offset_size_pair = compute_src_byte_offset_and_token_char_bytes<true>(
        d_str, -1, to_token_pos, info.token_size_sum - char_bytes_copied);
      if (dir == Dir::FORWARD) {
        thrust::copy(thrust::seq,
                     d_str.data() + offset_size_pair.first,
                     d_str.data() + offset_size_pair.first + offset_size_pair.second,
                     char_buf_ptr + char_bytes_copied);
        offset_buf_ptr[token_idx] = char_bytes_copied;
      } else {
        thrust::copy(thrust::seq,
                     d_str.data() + offset_size_pair.first,
                     d_str.data() + offset_size_pair.first + offset_size_pair.second,
                     char_buf_ptr);
        offset_buf_ptr[0] = 0;
      }
    }
    offset_buf_ptr[info.token_count] = info.token_size_sum;
  }
};

// Generic split function used by split_record and rsplit_record
template <typename TokenReader, typename TokenCopier>
contiguous_split_record_result contiguous_split_record_fn(strings_column_view const& strings,
                                                          TokenReader reader,
                                                          TokenCopier copier,
                                                          rmm::mr::device_memory_resource* mr,
                                                          cudaStream_t stream)
{
  // read each string element of the input column to count the number of tokens
  // and compute the memory offsets

  auto strings_count = strings.size();
  rmm::device_vector<size_type> d_token_counts(strings_count);
  rmm::device_vector<size_type> d_token_size_sums(strings_count);
  rmm::device_vector<size_type> d_memory_offsets(strings_count + 1);

  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(strings_count),
                    thrust::make_zip_iterator(thrust::make_tuple(
                      d_token_counts.begin(), d_token_size_sums.begin(), d_memory_offsets.begin())),
                    reader);

  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         d_memory_offsets.begin(),
                         d_memory_offsets.end(),
                         d_memory_offsets.begin());

  // allocate and copy

  thrust::host_vector<size_type> h_token_counts    = d_token_counts;
  thrust::host_vector<size_type> h_token_size_sums = d_token_size_sums;
  thrust::host_vector<size_type> h_memory_offsets  = d_memory_offsets;

  auto memory_size  = h_memory_offsets.back();
  auto all_data_ptr = std::make_unique<rmm::device_buffer>(memory_size, stream, mr);

  auto d_all_data_ptr        = reinterpret_cast<char*>(all_data_ptr->data());
  auto d_token_counts_ptr    = d_token_counts.data().get();
  auto d_memory_offsets_ptr  = d_memory_offsets.data().get();
  auto d_token_size_sums_ptr = d_token_size_sums.data().get();
  auto copy_info_begin       = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [d_all_data_ptr, d_token_counts_ptr, d_memory_offsets_ptr, d_token_size_sums_ptr] __device__(
      auto i) {
      return copy_info{i,
                       d_token_counts_ptr[i],
                       d_token_size_sums_ptr[i],
                       d_all_data_ptr + d_memory_offsets_ptr[i]};
    });

  thrust::for_each(
    rmm::exec_policy(stream)->on(stream), copy_info_begin, copy_info_begin + strings_count, copier);

  // update column_view objects

  std::vector<column_view> column_views{};
  for (size_type i = 0; i < strings_count; ++i) {
    if (h_token_counts[i] == 0) {
      column_views.emplace_back(strings.parent().type(), 0, nullptr);
    } else {
      auto memory_ptr    = d_all_data_ptr + h_memory_offsets[i];
      auto char_buf_size = cudf::util::round_up_safe(h_token_size_sums[i], split_align);

      auto char_buf_ptr = memory_ptr;
      memory_ptr += char_buf_size;
      auto offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

      column_views.emplace_back(
        strings.parent().type(),
        h_token_counts[i],
        nullptr,
        nullptr,
        UNKNOWN_NULL_COUNT,
        0,
        std::vector<column_view>{
          column_view(strings.offsets().type(), h_token_counts[i] + 1, offset_buf_ptr),
          column_view(strings.chars().type(), h_token_size_sums[i], char_buf_ptr)});
    }
  }

  CUDA_TRY(cudaStreamSynchronize(stream));

  return contiguous_split_record_result{std::move(column_views), std::move(all_data_ptr)};
}

}  // namespace

template <Dir dir>
contiguous_split_record_result contiguous_split_record(
  strings_column_view const& strings,
  string_scalar const& delimiter      = string_scalar(""),
  size_type maxsplit                  = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();
  auto has_validity    = strings.parent().nullable();

  auto d_strings_column_ptr = column_device_view::create(strings.parent(), stream);
  if (delimiter.size() == 0) {
    return contiguous_split_record_fn(
      strings,
      whitespace_token_reader_fn<dir>{*d_strings_column_ptr, max_tokens, has_validity},
      whitespace_token_copier_fn<dir>{*d_strings_column_ptr, has_validity},
      mr,
      stream);
  } else {
    string_view d_delimiter(delimiter.data(), delimiter.size());
    return contiguous_split_record_fn(
      strings,
      token_reader_fn<dir>{*d_strings_column_ptr, d_delimiter, max_tokens, has_validity},
      token_copier_fn<dir>{*d_strings_column_ptr, d_delimiter, has_validity},
      mr,
      stream);
  }
}

}  // namespace detail

// external APIs

contiguous_split_record_result contiguous_split_record(strings_column_view const& strings,
                                                       string_scalar const& delimiter,
                                                       size_type maxsplit,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contiguous_split_record<detail::Dir::FORWARD>(strings, delimiter, maxsplit, mr, 0);
}

contiguous_split_record_result contiguous_rsplit_record(strings_column_view const& strings,
                                                        string_scalar const& delimiter,
                                                        size_type maxsplit,
                                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contiguous_split_record<detail::Dir::BACKWARD>(
    strings, delimiter, maxsplit, mr, 0);
}

}  // namespace strings
}  // namespace cudf
