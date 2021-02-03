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

#include <strings/char_types/is_flags.h>
#include <strings/utilities.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/modify_strings.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {  // anonym.

// base class for 2-passes string modification:
// 1st pass: probing string manipulation memory load requirements
// 2nd pass: executing string modification.
//
struct probe_execute_base {
  using char_info = thrust::pair<uint32_t, detail::character_flags_table_type>;

  probe_execute_base(column_device_view const d_column,
                     int32_t const* d_offsets = nullptr,
                     char* d_chars            = nullptr)
    : d_column_(d_column),
      d_flags_(get_character_flags_table()),       // set flag table
      d_case_table_(get_character_cases_table()),  // set case table
      d_offsets_(d_offsets),
      d_chars_(d_chars)
  {
  }

  __host__ __device__ column_device_view const get_column(void) const { return d_column_; }

  __device__ char_info get_char_info(char_utf8 chr) const
  {
    uint32_t code_point                     = detail::utf8_to_codepoint(chr);
    detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;
    return char_info{code_point, flag};
  }

  __device__ char_utf8 convert_char(char_info const& info) const
  {
    return detail::codepoint_to_utf8(d_case_table_[info.first]);
  }

  __device__ char* get_output_ptr(size_type idx)
  {
    return d_chars_ && d_offsets_ ? d_chars_ + d_offsets_[idx] : nullptr;
  }

 private:
  column_device_view const d_column_;
  character_flags_table_type const* d_flags_;
  character_cases_table_type const* d_case_table_;
  int32_t const* d_offsets_;
  char* d_chars_;
};

// class that factors out the common inside-loop behavior
// of operator() between capitalize's `probe` and `execute`;
//(public inheritance to allow getters pass-through
// in derived classes);
//
struct probe_execute_capitalize : public probe_execute_base {
  explicit probe_execute_capitalize(column_device_view const d_column)
    : probe_execute_base(d_column)
  {
  }

  probe_execute_capitalize(column_device_view const d_column,
                           int32_t const* d_offsets,
                           char* d_chars)
    : probe_execute_base(d_column, d_offsets, d_chars)
  {
  }

  __device__ char_utf8 generate_chr(string_view::const_iterator itr, string_view d_str) const
  {
    auto the_chr = *itr;

    auto pair_char_info                     = get_char_info(the_chr);
    detail::character_flags_table_type flag = pair_char_info.second;

    if ((itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag))
      the_chr = convert_char(pair_char_info);

    return the_chr;
  }
};

// functor for probing string capitalization
// requirements:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct probe_capitalize : private probe_execute_capitalize {
  explicit probe_capitalize(column_device_view const d_column)
    :  // probe_execute_base(d_column)
      probe_execute_capitalize(d_column)
  {
  }

  __device__ int32_t operator()(size_type idx) const
  {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    int32_t bytes     = 0;

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      bytes += detail::bytes_in_char_utf8(generate_chr(itr, d_str));
    }
    return bytes;
  }
};

// functor for executing string capitalization:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct execute_capitalize : private probe_execute_capitalize {
  execute_capitalize(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    :  // probe_execute_base(d_column, d_offsets, d_chars)
      probe_execute_capitalize(d_column, d_offsets, d_chars)
  {
  }

  __device__ int32_t operator()(size_type idx)
  {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    char* d_buffer    = get_output_ptr(idx);

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      d_buffer += detail::from_char_utf8(generate_chr(itr, d_str), d_buffer);
    }
    return 0;
  }
};

// class that factors out the common inside-loop behavior
// of operator() between title's `probe` and `execute`;
//(public inheritance to allow getters pass-through
// in derived classes);
//
struct probe_execute_title : public probe_execute_base {
  explicit probe_execute_title(column_device_view const d_column) : probe_execute_base(d_column) {}

  probe_execute_title(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    : probe_execute_base(d_column, d_offsets, d_chars)
  {
  }

  __device__ thrust::pair<char_utf8, bool> generate_chr(string_view::const_iterator itr,
                                                        string_view d_str,
                                                        bool bcapnext) const
  {
    auto the_chr = *itr;

    auto pair_char_info                     = get_char_info(the_chr);
    detail::character_flags_table_type flag = pair_char_info.second;

    if (!IS_ALPHA(flag)) {
      bcapnext = true;
    } else {
      if (bcapnext ? IS_LOWER(flag) : IS_UPPER(flag)) the_chr = convert_char(pair_char_info);

      bcapnext = false;
    }

    return thrust::make_pair(the_chr, bcapnext);
  }
};

// functor for probing string title-ization
// requirements:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct probe_title : private probe_execute_title {
  explicit probe_title(column_device_view const d_column) : probe_execute_title(d_column) {}

  __device__ int32_t operator()(size_type idx) const
  {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    int32_t bytes     = 0;

    bool bcapnext = true;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto pair_char_flag = generate_chr(itr, d_str, bcapnext);
      bcapnext            = pair_char_flag.second;

      bytes += detail::bytes_in_char_utf8(pair_char_flag.first);
    }
    return bytes;
  }
};

// functor for executing string title-ization:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct execute_title : private probe_execute_title {
  execute_title(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    : probe_execute_title(d_column, d_offsets, d_chars)
  {
  }

  __device__ int32_t operator()(size_type idx)
  {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    char* d_buffer    = get_output_ptr(idx);

    bool bcapnext = true;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto pair_char_flag = generate_chr(itr, d_str, bcapnext);
      bcapnext            = pair_char_flag.second;

      d_buffer += detail::from_char_utf8(pair_char_flag.first, d_buffer);
    }
    return 0;
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<column> capitalize(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::modify_strings<detail::probe_capitalize, detail::execute_capitalize>(
    strings, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> title(strings_column_view const& strings,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::modify_strings<detail::probe_title, detail::execute_title>(
    strings, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
