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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <strings/utilities.cuh>
#include <strings/utilities.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {  // anonym.

//base class for probing string
//manipulation memory load requirements;
//and for executing string modification:
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
      d_chars_(d_chars) {}

  __host__ __device__ column_device_view const get_column(void) const { return d_column_; }

  __device__ char_info get_char_info(char_utf8 chr) const {
    uint32_t code_point                     = detail::utf8_to_codepoint(chr);
    detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;
    return char_info{code_point, flag};
  }

  __device__ char_utf8 convert_char(char_info const& info) const {
    return detail::codepoint_to_utf8(d_case_table_[info.first]);
  }

  __device__ char* get_output_ptr(size_type idx) {
    return d_chars_ && d_offsets_ ? d_chars_ + d_offsets_[idx] : nullptr;
  }

 private:
  column_device_view const d_column_;
  character_flags_table_type const* d_flags_;
  character_cases_table_type const* d_case_table_;
  int32_t const* d_offsets_;
  char* d_chars_;
};

//class that factors out the common inside-loop behavior
//of operator() between capitalize's `probe` and `execute`;
//(public inheritance to allow getters pass-through
//in derived classes);
//
struct probe_execute_capitalize : public probe_execute_base {
  explicit probe_execute_capitalize(column_device_view const d_column)
    : probe_execute_base(d_column) {}

  probe_execute_capitalize(column_device_view const d_column,
                           int32_t const* d_offsets,
                           char* d_chars)
    : probe_execute_base(d_column, d_offsets, d_chars) {}

  __device__ char_utf8 generate_chr(string_view::const_iterator itr, string_view d_str) const {
    auto the_chr = *itr;

    auto pair_char_info                     = get_char_info(the_chr);
    detail::character_flags_table_type flag = pair_char_info.second;

    if ((itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag))
      the_chr = convert_char(pair_char_info);

    return the_chr;
  }
};

//functor for probing string capitalization
//requirements:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct probe_capitalize : private probe_execute_capitalize {
  explicit probe_capitalize(column_device_view const d_column)
    :  //probe_execute_base(d_column)
      probe_execute_capitalize(d_column) {}

  __device__ int32_t operator()(size_type idx) const {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    int32_t bytes     = 0;

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      bytes += detail::bytes_in_char_utf8(generate_chr(itr, d_str));
    }
    return bytes;
  }
};

//functor for executing string capitalization:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct execute_capitalize : private probe_execute_capitalize {
  execute_capitalize(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    :  //probe_execute_base(d_column, d_offsets, d_chars)
      probe_execute_capitalize(d_column, d_offsets, d_chars) {}

  __device__ int32_t operator()(size_type idx) {
    if (get_column().is_null(idx)) return 0;  // null string

    string_view d_str = get_column().template element<string_view>(idx);
    char* d_buffer    = get_output_ptr(idx);

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      d_buffer += detail::from_char_utf8(generate_chr(itr, d_str), d_buffer);
    }
    return 0;
  }
};

//class that factors out the common inside-loop behavior
//of operator() between title's `probe` and `execute`;
//(public inheritance to allow getters pass-through
//in derived classes);
//
struct probe_execute_title : public probe_execute_base {
  explicit probe_execute_title(column_device_view const d_column) : probe_execute_base(d_column) {}

  probe_execute_title(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    : probe_execute_base(d_column, d_offsets, d_chars) {}

  __device__ thrust::pair<char_utf8, bool> generate_chr(string_view::const_iterator itr,
                                                        string_view d_str,
                                                        bool bcapnext) const {
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

//functor for probing string title-ization
//requirements:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct probe_title : private probe_execute_title {
  explicit probe_title(column_device_view const d_column) : probe_execute_title(d_column) {}

  __device__ int32_t operator()(size_type idx) const {
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

//functor for executing string title-ization:
//(private inheritance to prevent polymorphic use,
// a requirement that came up in code review)
//
struct execute_title : private probe_execute_title {
  execute_title(column_device_view const d_column, int32_t const* d_offsets, char* d_chars)
    : probe_execute_title(d_column, d_offsets, d_chars) {}

  __device__ int32_t operator()(size_type idx) {
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

template <typename device_probe_functor, typename device_execute_functor>
std::unique_ptr<column> modify_strings(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0) {
  auto strings_count = strings.size();
  if (strings_count == 0) return detail::make_empty_strings_column(mr, stream);

  auto execpol = rmm::exec_policy(stream);

  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  size_type null_count = strings.null_count();

  // copy null mask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);
  // get the lookup tables used for case conversion

  device_probe_functor d_probe_fctr{d_column};

  // build offsets column -- calculate the size of each output string
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), d_probe_fctr);
  auto offsets_column = detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto offsets_view  = offsets_column->view();
  auto d_new_offsets = offsets_view.template data<
    int32_t>();  //not sure why this requires `.template` and the next one (`d_chars = ...`) doesn't

  // build the chars column -- convert characters based on case_flag parameter
  size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

  device_execute_functor d_execute_fctr{d_column, d_new_offsets, d_chars};

  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     d_execute_fctr);

  //
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  //namespace detail

std::unique_ptr<column> capitalize(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr) {
  CUDF_FUNC_RANGE();
  return detail::modify_strings<detail::probe_capitalize, detail::execute_capitalize>(strings, mr);
}

std::unique_ptr<column> title(strings_column_view const& strings,
                              rmm::mr::device_memory_resource* mr) {
  CUDF_FUNC_RANGE();
  return detail::modify_strings<detail::probe_title, detail::execute_title>(strings, mr);
}

}  //namespace strings
}  //namespace cudf
