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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <strings/utilities.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <vector>
#include <thrust/transform.h>

namespace cudf
{
namespace strings
{
namespace detail
{

using string_index_pair = thrust::pair<const char*,size_type>;

namespace
{

/**
 * @brief Common token counter for all split methods in this file.
 */
struct token_counter_fn
{
    column_device_view const d_strings;
    string_view const d_delimiter;
    size_type tokens;

    // returns the number of possible tokens in each string
    __device__ size_type operator()(size_type idx) const
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        if( d_str.empty() )
            return 1;
        size_type delim_count = 0;
        auto delim_length = d_delimiter.length();
        auto pos = d_str.find(d_delimiter);
        while(pos >= 0)
        {
            ++delim_count;
            pos = d_str.find(d_delimiter, pos + delim_length);
        }
        size_type rtn = delim_count + 1;
        if((tokens > 0) && (rtn > tokens))
            rtn = tokens;
        return rtn;
    }
};

//
// This will create new columns by splitting the array of strings vertically.
// All the first tokens go in the first column, all the second tokens go in the second column, etc.
// It is comparable to Pandas split with expand=True but the rows/columns are transposed.
// Example:
//   import pandas as pd
//   pd_series = pd.Series(['', None, 'a_b', '_a_b_', '__aa__bb__', '_a__bbb___c', '_aa_b__ccc__'])
//   print(pd_series.str.split(pat='_', expand=True))
//            0     1     2     3     4     5     6
//      0    ''  None  None  None  None  None  None
//      1  None  None  None  None  None  None  None
//      2     a     b  None  None  None  None  None
//      3    ''     a     b    ''  None  None  None
//      4    ''    ''    aa    ''    bb    ''    ''
//      5    ''     a    ''   bbb    ''    ''     c
//      6    ''    aa     b    ''   ccc    ''    ''
//
//   print(pd_series.str.split(pat='_', n=1, expand=True))
//            0            1
//      0    ''         None
//      1  None         None
//      2     a            b
//      3    ''         a_b_
//      4    ''    _aa__bb__
//      5    ''   a__bbb___c
//      6    ''  aa_b__ccc__
//
//   print(pd_series.str.split(pat='_', n=2, expand=True))
//            0     1         2
//      0    ''  None      None
//      1  None  None      None
//      2     a     b      None
//      3    ''     a        b_
//      4    ''        aa__bb__
//      5    ''     a  _bbb___c
//      6    ''    aa  b__ccc__
//
struct split_tokenizer_fn
{
    column_device_view const d_strings;  // strings to split
    string_view const d_delimiter;       // delimiter for split

    __device__ string_index_pair operator()(size_type idx,
                                            size_type col_idx, size_type column_count,
                                            size_type const* d_token_counts) const
    {
        // token_count already includes the max-split value
        size_type token_count = d_token_counts[idx];
        if( col_idx >= token_count || d_strings.is_null(idx) )
            return string_index_pair{nullptr,0};
        string_view d_str = d_strings.element<string_view>(idx);
        auto delim_nchars = d_delimiter.length();
        size_type spos = 0;
        size_type nchars = d_str.length();
        size_type epos = nchars;
        // skip delimiters until we reach the col_idx or the token_count
        for( size_type c=0; c < (token_count-1); ++c )
        {
            epos = d_str.find(d_delimiter,spos);
            if( c==col_idx )  // found our column
                break;
            spos = epos + delim_nchars;
            epos = nchars;
        }
        // this will be the string for this column
        string_index_pair result{d_str.data(),0}; // init to empty string
        if( spos < epos )
        {
            spos = d_str.byte_offset(spos); // convert character pos
            epos = d_str.byte_offset(epos); // to byte offset
            result = string_index_pair{ d_str.data() + spos, (epos-spos) };
        }
        return result;
    }
};

/**
 * @brief Extracts a specific set of tokens from a strings column.
 *
 * This will perform the split starting at the end of each string.
 */
struct rsplit_tokenizer_fn
{
    column_device_view const d_strings;  // strings to split
    string_view const d_delimiter;       // delimiter for split

    __device__ string_index_pair operator()(size_type idx,
                                            size_type col_idx, size_type column_count,
                                            size_type const* d_token_counts) const
    {
        // token_count already includes the max-split value
        size_type token_count = d_token_counts[idx];
        if( col_idx >= token_count || d_strings.is_null(idx) )
            return string_index_pair{nullptr,0};
        string_view d_str = d_strings.element<string_view>(idx);
        auto delim_nchars = d_delimiter.length();
        size_type spos = 0;
        size_type nchars = d_str.length();
        size_type epos = nchars;
        // skip delimiters until we reach col-idx or token_count
        for( auto c=(token_count-1); c > 0; --c )
        {
            spos = d_str.rfind(d_delimiter,0,epos);
            if( c==col_idx ) // found our column
            {
                spos += delim_nchars;  // do not include delimiter
                break;
            }
            epos = spos;
            spos = 0;
        }
        // this will be the string for this column
        string_index_pair result{d_str.data(),0}; // init to empty string
        if( spos < epos )
        {
            spos = d_str.byte_offset(spos); // convert char pos
            epos = d_str.byte_offset(epos); // to byte offset
            result = string_index_pair{ d_str.data() + spos, (epos-spos) };
        }
        return result;
    }
};

/**
 * @brief Special-case token counter for whitespace delimiter.
 *
 * Leading and trailing and duplicate delimiters are ignored.
 */
struct whitespace_token_counter_fn
{
    column_device_view const d_strings;
    size_type tokens; // maximum number of tokens

    // count the 'words' only between non-whitespace characters
    __device__ size_type operator()(size_type idx) const
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        size_type dcount = 0;
        bool spaces = true; // need to treat a run of whitespace as a single delimiter
        auto itr = d_str.begin();
        while( itr != d_str.end() )
        {
            char_utf8 ch = *itr;
            if( spaces == (ch <= ' ') )
                itr++;
            else
            {
                dcount += static_cast<size_type>(spaces);
                spaces = !spaces;
            }
        }
        if( tokens && (dcount > tokens) )
            dcount = tokens;
        if( dcount==0 )
            dcount = 1; // always allow empty string
        return dcount;
    }
};

//
// This is the whitespace-delimiter version of the column split function.
// Like the one above, it can be compared to Pandas split with expand=True but
// with the rows/columns transposed.
//
//  import pandas as pd
//  pd_series = pd.Series(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])
//  print(pd_series.str.split(pat=None, expand=True))
//            0     1     2
//      0  None  None  None
//      1  None  None  None
//      2     a     b  None
//      3     a     b  None
//      4    aa    bb  None
//      5     a   bbb     c
//      6    aa     b   ccc
//
//  print(pd_series.str.split(pat=None, n=1, expand=True))
//            0         1
//      0  None      None
//      1  None      None
//      2     a         b
//      3     a        b
//      4    aa      bb
//      5     a   bbb   c
//      6    aa  b  ccc
//
//  print(pd_series.str.split(pat=None, n=2, expand=True))
//            0     1      2
//      0  None  None   None
//      1  None  None   None
//      2     a     b   None
//      3     a     b   None
//      4    aa    bb   None
//      5     a   bbb      c
//      6    aa     b  ccc
//
// Like the split_record method, there are no empty strings here.
//
struct whitespace_split_tokenizer_fn
{
    column_device_view const d_strings;  // strings to split
    size_type tokens;                    // maximum number of tokens

    __device__ string_index_pair operator()(size_type idx,
                                            size_type col_idx, size_type column_count,
                                            size_type const* d_token_counts) const
    {
        size_type token_count = d_token_counts[idx];
        if( col_idx >= token_count || d_strings.is_null(idx) )
            return string_index_pair{nullptr,0};
        string_view d_str = d_strings.element<string_view>(idx);
        size_type c = 0;
        size_type nchars = d_str.length();
        size_type spos = 0;
        size_type epos = nchars;
        bool spaces = true;  // need to treat a run of whitespace as a single delimiter
        for( size_type pos=0; pos < nchars; ++pos )
        {
            char_utf8 ch = d_str[pos];
            if( spaces == (ch <= ' ') )
            {
                if( spaces )
                    spos = pos+1;
                else
                    epos = pos+1;
                continue;
            }
            if( !spaces )
            {
                epos = nchars;
                if( (c+1)==tokens ) // hit max tokens
                    break;
                epos = pos;
                if( c==col_idx ) // found our column
                    break;
                spos = pos+1;
                epos = nchars;
                ++c;
            }
            spaces = !spaces;
        }
        // this is the string for this column
        string_index_pair result{nullptr,0}; // init to null string
        if( spos < epos )
        {
            spos = d_str.byte_offset(spos); // convert char pos
            epos = d_str.byte_offset(epos); // to byte offset
            result = string_index_pair{ d_str.data() + spos, (epos-spos) };
        }
        return result;
    }
};

/**
 * @brief Extracts a specific set of tokens from a strings column
 * using whitespace as delimiter but splitting starts from the end
 * of each string.
 */
struct whitespace_rsplit_tokenizer_fn
{
    column_device_view const d_strings;  // strings to split
    size_type tokens;                    // maximum number of tokens

    __device__ string_index_pair operator()(size_type idx,
                                            size_type col_idx, size_type column_count,
                                            size_type const* d_token_counts) const
    {
        size_type token_count = d_token_counts[idx];
        if( col_idx >= token_count || d_strings.is_null(idx) )
            return string_index_pair{nullptr,0};
        string_view d_str = d_strings.element<string_view>(idx);
        size_type c = (token_count-1);
        size_type nchars = d_str.length();
        size_type spos = 0;
        size_type epos = nchars;
        bool spaces = true;  // need to treat a run of whitespace as a single delimiter
        for( int pos=nchars; pos > 0; --pos )
        {
            char_utf8 ch = d_str[pos-1];
            if( spaces == (ch <= ' ') )
            {
                if( spaces )
                    epos = pos-1;
                else
                    spos = pos-1;
                continue;
            }
            if( !spaces )
            {
                spos = 0;
                if( (column_count-c)==tokens )  // hit max tokens
                    break;
                spos = pos;
                if( c==col_idx ) // found our column
                    break;
                epos = pos-1;
                spos = 0;
                --c;
            }
            spaces = !spaces;
        }
        // this is the string for this column
        string_index_pair result{nullptr,0}; // init to null string
        if( spos < epos )
        {
            spos = d_str.byte_offset(spos); // convert char pos
            epos = d_str.byte_offset(epos); // to byte offset
            result = string_index_pair{ d_str.data() + spos, (epos-spos) };
        }
        return result;
    }
};

// TODO: copied from cuDF PR 3685 (should include instead of duplicating once
// PR 3685 is merged)
#if 1
template <typename T>
__device__ inline T round_up_pow2(T number_to_round, T modulus) {
    return (number_to_round + (modulus - 1)) & -modulus;
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_type split_align = 64;
#endif

__device__ size_type compute_memory_size(
    size_type token_count, size_type token_size_sum) {
  return round_up_pow2(token_size_sum, split_align) +
    round_up_pow2(
      (token_count + 1) * static_cast<size_type>(sizeof(size_type)),
      split_align);
}

/**
 * @brief Compute the number of tokens, the total byte sizes of the tokens, and
 * required memory size for the `idx'th` string element of `d_strings`.
 */
template <bool forward>
struct token_reader_fn {
  column_device_view const d_strings;  // strings to split
  string_view const d_delimiter;  // delimiter for split
  size_type const max_tokens = std::numeric_limits<size_type>::max();
  bool const has_validity = false;

  template <bool last>
  __device__ size_type compute_token_char_bytes(
      string_view const& d_str,
      size_type start_pos, size_type end_pos, size_type delimiter_pos) const {
    if (last) {
       return forward ?
         d_str.byte_offset(end_pos) - d_str.byte_offset(start_pos) :
         d_str.byte_offset(end_pos);
    }
    else {
      return forward ?
        d_str.byte_offset(delimiter_pos) - d_str.byte_offset(start_pos) :
        d_str.byte_offset(end_pos) -
          d_str.byte_offset(delimiter_pos + d_delimiter.length());
    }
  }

  // returns a tuple of token count, sum of token sizes in bytes, and required
  // memory block size
  __device__ thrust::tuple<size_type, size_type, size_type>
  operator()(size_type idx) const {
    if (has_validity && d_strings.is_null(idx)) {
      return thrust::make_tuple<size_type, size_type, size_type>(0, 0, 0);
    }

    auto const d_str = d_strings.element<string_view>(idx);
    size_type token_count = 0;
    size_type token_size_sum = 0;
    size_type start_pos = 0;  // updates only if forward is true
    auto end_pos = d_str.length();  // updates only if forward is false
    while (token_count < max_tokens - 1) {
      auto const delimiter_pos =
        forward ? d_str.find(d_delimiter, start_pos) :
                  d_str.rfind(d_delimiter, start_pos, end_pos);
      if (delimiter_pos != -1) {
        token_count++;
        token_size_sum += compute_token_char_bytes<false>(
                            d_str, start_pos, end_pos, delimiter_pos);
        if (forward) {
          start_pos = delimiter_pos + d_delimiter.length();
        }
        else {
          end_pos = delimiter_pos;
        }
      }
      else {
        break;
      }
    }
    token_count++;
    token_size_sum +=
      compute_token_char_bytes<true>(d_str, start_pos, end_pos, -1);

    auto const memory_size = compute_memory_size(token_count, token_size_sum);

    return thrust::make_tuple<size_type, size_type, size_type>(
      token_count, token_size_sum, memory_size);
  }
};

/**
 * @brief Copy the tokens from the `idx'th` string element of `d_strings` to
 * the contiguous memory buffer.
 */
template <bool forward>
struct token_copier_fn {
  column_device_view const d_strings;  // strings to split
  string_view const d_delimiter;  // delimiter for split
  bool const has_validity = false;

  template <bool last>
  __device__ thrust::pair<size_type, size_type>
  compute_src_byte_offset_and_token_char_bytes(
      string_view const& d_str,
      size_type start_pos, size_type end_pos, size_type delimiter_pos) const {
    if (last) {
      auto const src_byte_offset = forward ? d_str.byte_offset(start_pos) : 0;
      auto const token_char_bytes =
        forward ?
          d_str.byte_offset(end_pos) - src_byte_offset :
          d_str.byte_offset(end_pos);
      return thrust::make_pair<size_type, size_type>(
        src_byte_offset, token_char_bytes);
    }
    else {
      auto const src_byte_offset =
        forward ? d_str.byte_offset(start_pos) :
                  d_str.byte_offset(delimiter_pos + d_delimiter.length());
      auto const token_char_bytes =
        forward ?
          d_str.byte_offset(delimiter_pos) - src_byte_offset :
          d_str.byte_offset(end_pos) - src_byte_offset;
      return thrust::make_pair<size_type, size_type>(
        src_byte_offset, token_char_bytes);
    }
  }

  __device__ void
  operator()(
      thrust::tuple<size_type, size_type, void*, size_type>
        idx_token_count_memory_ptr_token_size_sum) const {
    auto const idx = idx_token_count_memory_ptr_token_size_sum.get<0>();
    auto const token_count = idx_token_count_memory_ptr_token_size_sum.get<1>();

    if (token_count == 0) {
      return;
    }

    auto memory_ptr =
      static_cast<char*>(idx_token_count_memory_ptr_token_size_sum.get<2>());
    auto const token_size_sum =
      idx_token_count_memory_ptr_token_size_sum.get<3>();

    auto const char_buf_size = round_up_pow2(token_size_sum, split_align);
    auto const char_buf_ptr = memory_ptr;
    memory_ptr += char_buf_size;
    auto const offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

    auto const d_str = d_strings.element<string_view>(idx);
    size_type token_idx = 0;
    size_type char_bytes_copied = 0;
    size_type start_pos = 0;  // updates only if forward is true
    auto end_pos = d_str.length();  // updates only if forward is false
    while (token_idx < token_count - 1) {
      auto const delimiter_pos =
        forward ? d_str.find(d_delimiter, start_pos) :
                  d_str.rfind(d_delimiter, start_pos, end_pos);
      if (delimiter_pos != -1) {
        auto const offset_size_pair =
          compute_src_byte_offset_and_token_char_bytes<false>(
            d_str, start_pos, end_pos, delimiter_pos);
        if (forward) {
          thrust::copy(
            thrust::seq,
            d_str.data() + offset_size_pair.first,
            d_str.data() + offset_size_pair.first + offset_size_pair.second,
            char_buf_ptr + char_bytes_copied);
          offset_buf_ptr[token_idx] = char_bytes_copied;
        }
        else {
          auto const char_buf_offset =
            token_size_sum - char_bytes_copied - offset_size_pair.second;
          thrust::copy(
            thrust::seq,
            d_str.data() + offset_size_pair.first,
            d_str.data() + offset_size_pair.first + offset_size_pair.second,
            char_buf_ptr + char_buf_offset);
          offset_buf_ptr[token_count - 1 - token_idx] = char_buf_offset;
        }
        token_idx++;
        char_bytes_copied += offset_size_pair.second;
        if (forward) {
          start_pos = delimiter_pos + d_delimiter.length();
        }
        else {
          end_pos = delimiter_pos;
        }
      }
      else {
        break;
      }
    }

    auto const offset_size_pair =
      compute_src_byte_offset_and_token_char_bytes<true>(
        d_str, start_pos, end_pos, -1);
    if (forward) {
      thrust::copy(
        thrust::seq,
        d_str.data() + offset_size_pair.first,
        d_str.data() + offset_size_pair.first + offset_size_pair.second,
        char_buf_ptr + char_bytes_copied);
      offset_buf_ptr[token_idx] = char_bytes_copied;
    }
    else {
      thrust::copy(thrust::seq,
                   d_str.data(), d_str.data() + offset_size_pair.second,
                   char_buf_ptr);
      offset_buf_ptr[0] = 0;
    }
    offset_buf_ptr[token_count] = token_size_sum;
  }
};

/**
 * @brief Compute the number of tokens, the total byte sizes of the tokens, and
 * required memory size for the `idx'th` string element of `d_strings`.
 */
template <bool forward>
struct whitespace_token_reader_fn {
  column_device_view const d_strings;  // strings to split
  size_type const max_tokens = std::numeric_limits<size_type>::max();
  bool const has_validity = false;

  template <bool last>
  __device__ size_type compute_token_char_bytes(
      string_view const& d_str,
      size_type cur_pos, size_type to_token_pos) const {
    if (last) {
      return forward ?
        d_str.byte_offset(d_str.length()) - d_str.byte_offset(to_token_pos) :
        d_str.byte_offset(to_token_pos + 1) - d_str.byte_offset(0);
    }
    else {
      return forward ?
        d_str.byte_offset(cur_pos) - d_str.byte_offset(to_token_pos) :
        d_str.byte_offset(to_token_pos + 1) - d_str.byte_offset(cur_pos + 1);
    }
  }

  __device__ thrust::tuple<size_type, size_type, size_type>
  operator()(size_type idx) const {
    if (has_validity && d_strings.is_null(idx)) {
      return thrust::make_tuple<size_type, size_type, size_type>(0, 0, 0);
    }

    auto const d_str = d_strings.element<string_view>(idx);
    size_type token_count = 0;
    size_type token_size_sum = 0;
    auto spaces = true;
    auto reached_max_tokens = false;
    size_type to_token_pos = 0;
    for (size_type i = 0; i < d_str.length(); ++i) {
      auto const cur_pos = forward ? i : d_str.length() - 1 - i;
      auto const ch = d_str[cur_pos];
      if (spaces != (ch <= ' ')) {
        if (spaces) {  // from whitespace(s) to a new token
          to_token_pos = cur_pos;
        }
        else {  // from a token to whiltespace(s)
          if (token_count < max_tokens - 1) {
            token_count++;
            token_size_sum +=
              compute_token_char_bytes<false>(d_str, cur_pos, to_token_pos);
          }
          else {
            reached_max_tokens = true;
            break;
          }
        }
        spaces = !spaces;
      }
    }
    if (reached_max_tokens || !spaces) {
      token_count++;
      token_size_sum +=
        compute_token_char_bytes<true>(d_str, -1, to_token_pos);
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
template <bool forward>
struct whitespace_token_copier_fn {
  column_device_view const d_strings;  // strings to split
  bool const has_validity = false;

  template <bool last>
  __device__ thrust::pair<size_type, size_type>
  compute_src_byte_offset_and_token_char_bytes(
      string_view const& d_str,
      size_type cur_pos, size_type to_token_pos,
      size_type remaining_bytes) const {
    if (last) {
      auto const token_char_bytes = remaining_bytes;
      auto const src_byte_offset =
        forward ? d_str.byte_offset(to_token_pos) :
                  d_str.byte_offset(to_token_pos + 1) - token_char_bytes;
      return thrust::make_pair<size_type, size_type>(
        src_byte_offset, token_char_bytes);
    }
    else {
      auto const src_byte_offset =
        forward ? d_str.byte_offset(to_token_pos) :
          d_str.byte_offset(cur_pos + 1);
      auto const token_char_bytes =
        forward ?
          d_str.byte_offset(cur_pos) - src_byte_offset :
          d_str.byte_offset(to_token_pos + 1) - src_byte_offset;
      return thrust::make_pair<size_type, size_type>(
        src_byte_offset, token_char_bytes);
    }
  }

  __device__ void
  operator()(
      thrust::tuple<size_type, size_type, void*, size_type>
        idx_token_count_memory_ptr_token_size_sum) const {
    auto const idx = idx_token_count_memory_ptr_token_size_sum.get<0>();
    auto const token_count = idx_token_count_memory_ptr_token_size_sum.get<1>();

    if (token_count == 0) {
      return;
    }

    auto memory_ptr =
      static_cast<char*>(idx_token_count_memory_ptr_token_size_sum.get<2>());
    auto const token_size_sum =
      idx_token_count_memory_ptr_token_size_sum.get<3>();

    auto const char_buf_size = round_up_pow2(token_size_sum, split_align);
    auto const char_buf_ptr = memory_ptr;
    memory_ptr += char_buf_size;
    auto const offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

    auto const d_str = d_strings.element<string_view>(idx);
    size_type token_idx = 0;
    size_type char_bytes_copied = 0;
    auto spaces = true;
    size_type to_token_pos = 0;
    for (size_type i = 0; i < d_str.length(); ++i) {
      auto const cur_pos = forward ? i : d_str.length() - 1 - i;
      auto const ch = d_str[cur_pos];
      if (spaces != (ch <= ' ')) {
        if (spaces) {  // from whitespace(s) to a new token
          to_token_pos = cur_pos;
        }
        else {  // from a token to whiltespace(s)
          if (token_idx < token_count - 1) {
            auto const offset_size_pair =
              compute_src_byte_offset_and_token_char_bytes<false>(
                d_str, cur_pos, to_token_pos,
                token_size_sum - char_bytes_copied);
            if (forward) {
              thrust::copy(
                thrust::seq,
                d_str.data() + offset_size_pair.first,
                d_str.data() + offset_size_pair.first + offset_size_pair.second,
                char_buf_ptr + char_bytes_copied);
              offset_buf_ptr[token_idx] = char_bytes_copied;
            }
            else {
              auto const char_buf_offset =
                token_size_sum - char_bytes_copied - offset_size_pair.second;
              thrust::copy(
                thrust::seq,
                d_str.data() + offset_size_pair.first,
                d_str.data() + offset_size_pair.first + offset_size_pair.second,
                char_buf_ptr + char_buf_offset);
              offset_buf_ptr[token_count - 1 - token_idx] = char_buf_offset;
            }
            token_idx++;
            char_bytes_copied += offset_size_pair.second;
          }
          else {
            break;
          }
        }
        spaces = !spaces;
      }
    }
    if (token_idx < token_count) {
      auto const offset_size_pair =
        compute_src_byte_offset_and_token_char_bytes<true>(
          d_str, -1, to_token_pos, token_size_sum - char_bytes_copied);
      if (forward) {
        thrust::copy(
          thrust::seq,
          d_str.data() + offset_size_pair.first,
          d_str.data() + offset_size_pair.first + offset_size_pair.second,
          char_buf_ptr + char_bytes_copied);
        offset_buf_ptr[token_idx] = char_bytes_copied;
      }
      else {
        thrust::copy(
          thrust::seq,
          d_str.data() + offset_size_pair.first,
          d_str.data() + offset_size_pair.first + offset_size_pair.second,
          char_buf_ptr);
        offset_buf_ptr[0] = 0;
      }
    }
    offset_buf_ptr[token_count] = token_size_sum;
  }
};

// Generic split function used by split and rsplit
template<typename TokenCounter, typename Tokenizer>
std::unique_ptr<experimental::table> split_fn( size_type strings_count,
                                               TokenCounter counter,
                                               Tokenizer tokenizer,
                                               rmm::mr::device_memory_resource* mr,
                                               cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    // compute the number of tokens per string
    size_type columns_count = 0;
    rmm::device_vector<size_type> token_counts(strings_count);
    auto d_token_counts = token_counts.data().get();
    if( strings_count > 0 )
    {
        thrust::transform( execpol->on(stream),
            thrust::make_counting_iterator<size_type>(0),
            thrust::make_counting_iterator<size_type>(strings_count),
            d_token_counts, counter );
        // column count is the maximum number of tokens for any string
        columns_count = *thrust::max_element(execpol->on(stream),
                                             token_counts.begin(), token_counts.end() );
    }
    std::vector<std::unique_ptr<column>> results;
    // boundary case: if no columns, return one null column (issue #119)
    if( columns_count==0 )
    {
        results.push_back(std::make_unique<column>( data_type{STRING}, strings_count,
                          rmm::device_buffer{0,stream,mr}, // no data
                          create_null_mask(strings_count, ALL_NULL, stream, mr), strings_count ));
    }

    // Create each column.
    // Build a vector of pair<char*,int>'s' for each column.
    // Each pair points to a string for this column for each row.
    // Create the strings column using the strings factory.
    for( size_type col=0; col < columns_count; ++col )
    {
        rmm::device_vector<string_index_pair> indexes(strings_count);
        string_index_pair* d_indexes = indexes.data().get();
        thrust::transform(execpol->on(stream),
            thrust::make_counting_iterator<size_type>(0),
            thrust::make_counting_iterator<size_type>(strings_count),
            d_indexes,
            [tokenizer, col, columns_count, d_token_counts] __device__ (size_type idx) {
                return tokenizer(idx,col,columns_count, d_token_counts);
            });
        auto column = make_strings_column(indexes,stream,mr);
        results.emplace_back(std::move(column));
    }
    return std::make_unique<experimental::table>(std::move(results));
}

// Generic split function used by split_record and rsplit_record
template<typename TokenReader, typename TokenCopier>
contiguous_split_record_result contiguous_split_record_fn(
    strings_column_view const& strings,
    TokenReader reader,
    TokenCopier copier,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
  // read each string element of the input column to count the number of tokens
  // and compute the memory offsets

  auto strings_count = strings.size();
  rmm::device_vector<size_type> d_token_counts(strings_count);
  rmm::device_vector<size_type> d_token_size_sums(strings_count);
  rmm::device_vector<size_type> d_memory_offsets(strings_count + 1);

  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(strings_count),
    thrust::make_zip_iterator(
      thrust::make_tuple(
        d_token_counts.begin(), d_token_size_sums.begin(),
        d_memory_offsets.begin())),
    reader);

  thrust::exclusive_scan(
    rmm::exec_policy(stream)->on(stream),
    d_memory_offsets.begin(),
    d_memory_offsets.end(),
    d_memory_offsets.begin());

  // allocate and copy

  thrust::host_vector<size_type> h_token_counts = d_token_counts;
  thrust::host_vector<size_type> h_token_size_sums = d_token_size_sums;
  thrust::host_vector<size_type> h_memory_offsets = d_memory_offsets;

  auto memory_size = h_memory_offsets.back();
  auto all_data_ptr =
    std::make_unique<rmm::device_buffer>(memory_size, stream, mr);

  auto d_all_data_ptr = reinterpret_cast<char*>(all_data_ptr->data());
  auto d_token_counts_ptr = d_token_counts.data().get();
  auto d_memory_offsets_ptr = d_memory_offsets.data().get();
  auto d_token_size_sums_ptr = d_token_size_sums.data().get();
  auto idx_token_count_memory_ptr_token_size_sum_begin =
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [d_all_data_ptr, d_token_counts_ptr, d_memory_offsets_ptr,
       d_token_size_sums_ptr] __device__ (auto i) {
        return thrust::make_tuple<size_type, size_type, void*, size_type>(
          i, d_token_counts_ptr[i], d_all_data_ptr + d_memory_offsets_ptr[i],
          d_token_size_sums_ptr[i]);
      });

  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    idx_token_count_memory_ptr_token_size_sum_begin,
    idx_token_count_memory_ptr_token_size_sum_begin + strings_count,
    copier);

  // update column_view objects

  std::vector<column_view> column_views{};
  for (size_type i = 0; i < strings_count; ++i) {
    if (h_token_counts[i] == 0) {
      column_views.emplace_back(strings.parent().type(), 0, nullptr);
    }
    else {
      auto memory_ptr =
        d_all_data_ptr + h_memory_offsets[i];
      auto char_buf_size =
        cudf::util::round_up_safe(h_token_size_sums[i], split_align);

      auto char_buf_ptr = memory_ptr;
      memory_ptr += char_buf_size;
      auto offset_buf_ptr = reinterpret_cast<size_type*>(memory_ptr);

      column_views.emplace_back(
        strings.parent().type(),
        h_token_counts[i], nullptr, nullptr, UNKNOWN_NULL_COUNT, 0,
        std::vector<column_view>{
          column_view(
            strings.offsets().type(), h_token_counts[i] + 1, offset_buf_ptr),
          column_view(
            strings.chars().type(), h_token_size_sums[i], char_buf_ptr)});
    }
  }

  CUDA_TRY(cudaStreamSynchronize(stream));

  return contiguous_split_record_result{std::move(column_views),
                                        std::move(all_data_ptr)};
}

} // namespace


std::unique_ptr<experimental::table> split( strings_column_view const& strings,
                                            string_scalar const& delimiter = string_scalar(""),
                                            size_type maxsplit=-1,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");

    size_type max_tokens = 0;
    if( maxsplit > 0 )
        max_tokens = maxsplit + 1; // makes consistent with Pandas

    auto strings_column = column_device_view::create(strings.parent(),stream);
    if( delimiter.size()==0 )
    {
        return split_fn( strings.size(),
                         whitespace_token_counter_fn{*strings_column,max_tokens},
                         whitespace_split_tokenizer_fn{*strings_column,max_tokens},
                         mr, stream);
    }

    string_view d_delimiter( delimiter.data(), delimiter.size() );
    return split_fn( strings.size(),
                     token_counter_fn{*strings_column,d_delimiter,max_tokens},
                     split_tokenizer_fn{*strings_column,d_delimiter},
                     mr, stream);
}

std::unique_ptr<experimental::table> rsplit( strings_column_view const& strings,
                                             string_scalar const& delimiter = string_scalar(""),
                                             size_type maxsplit=-1,
                                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                             cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");

    size_type max_tokens = 0;
    if( maxsplit > 0 )
        max_tokens = maxsplit + 1; // makes consistent with Pandas

    auto strings_column = column_device_view::create(strings.parent(),stream);
    if( delimiter.size()==0 )
    {
        return split_fn( strings.size(),
                         whitespace_token_counter_fn{*strings_column,max_tokens},
                         whitespace_rsplit_tokenizer_fn{*strings_column,max_tokens},
                         mr, stream);
    }

    string_view d_delimiter( delimiter.data(), delimiter.size() );
    return split_fn( strings.size(),
                     token_counter_fn{*strings_column,d_delimiter,max_tokens},
                     rsplit_tokenizer_fn{*strings_column,d_delimiter},
                     mr, stream);
}

contiguous_split_record_result contiguous_split_record(
    strings_column_view const& strings,
    string_scalar const& delimiter = string_scalar(""),
    size_type maxsplit = -1,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0) {
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 :
                                        std::numeric_limits<size_type>::max();
  auto has_validity = strings.parent().nullable();

  auto d_strings_column_ptr =
    column_device_view::create(strings.parent(), stream);
  if (delimiter.size() == 0) {
    return contiguous_split_record_fn(
      strings,
      whitespace_token_reader_fn<true>{
        *d_strings_column_ptr, max_tokens, has_validity},
      whitespace_token_copier_fn<true>{*d_strings_column_ptr, has_validity},
      mr, stream);
  }
  else {
    string_view d_delimiter(delimiter.data(), delimiter.size());
    return contiguous_split_record_fn(
      strings,
      token_reader_fn<true>{
        *d_strings_column_ptr, d_delimiter, max_tokens, has_validity},
      token_copier_fn<true>{*d_strings_column_ptr, d_delimiter, has_validity},
      mr, stream);
  }
}

contiguous_split_record_result contiguous_rsplit_record(
    strings_column_view const& strings,
    string_scalar const& delimiter = string_scalar(""),
    size_type maxsplit = -1,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0) {
  CUDF_EXPECTS(delimiter.is_valid(), "Parameter delimiter must be valid");

  size_type max_tokens = std::numeric_limits<size_type>::max();
  if (maxsplit > 0) {
    max_tokens = maxsplit + 1;  // makes consistent with Pandas
  }
  auto has_validity = strings.parent().nullable();

  auto d_strings_column_ptr =
    column_device_view::create(strings.parent(),stream);
  if (delimiter.size() == 0) {
    return contiguous_split_record_fn(
      strings,
      whitespace_token_reader_fn<false>{
        *d_strings_column_ptr, max_tokens, has_validity},
      whitespace_token_copier_fn<false>{*d_strings_column_ptr, has_validity},
      mr, stream);
  }
  else {
    string_view d_delimiter(delimiter.data(), delimiter.size());
    return contiguous_split_record_fn(
      strings,
      token_reader_fn<false>{
        *d_strings_column_ptr, d_delimiter, max_tokens, has_validity},
      token_copier_fn<false>{*d_strings_column_ptr, d_delimiter, has_validity},
      mr, stream);
  }
}

} // namespace detail

// external APIs

std::unique_ptr<experimental::table> split( strings_column_view const& strings,
                                            string_scalar const& delimiter,
                                            size_type maxsplit,
                                            rmm::mr::device_memory_resource* mr )
{
    return detail::split( strings, delimiter, maxsplit, mr );
}

std::unique_ptr<experimental::table> rsplit( strings_column_view const& strings,
                                             string_scalar const& delimiter,
                                             size_type maxsplit,
                                             rmm::mr::device_memory_resource* mr)
{
    return detail::rsplit( strings, delimiter, maxsplit, mr );
}

contiguous_split_record_result contiguous_split_record(
    strings_column_view const& strings,
    string_scalar const& delimiter,
    size_type maxsplit,
    rmm::mr::device_memory_resource* mr) {
  return detail::contiguous_split_record(strings, delimiter, maxsplit, mr, 0);
}

contiguous_split_record_result contiguous_rsplit_record(
    strings_column_view const& strings,
    string_scalar const& delimiter,
    size_type maxsplit,
    rmm::mr::device_memory_resource* mr) {
  return detail::contiguous_rsplit_record(strings, delimiter, maxsplit, mr, 0);
}

} // namespace strings
} // namespace cudf
