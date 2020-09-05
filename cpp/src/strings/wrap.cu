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

namespace cudf {
namespace strings {
namespace detail {
namespace {  // anonym.

// execute string wrap:
//
struct execute_wrap {
  execute_wrap(column_device_view const d_column,
               int32_t const* d_offsets,
               char* d_chars,
               size_type width)
    : d_column_(d_column), d_offsets_(d_offsets), d_chars_(d_chars), width_(width)
  {
  }

  __device__ int32_t operator()(size_type idx)
  {
    if (d_column_.is_null(idx)) return 0;  // null string

    string_view d_str = d_column_.template element<string_view>(idx);
    char* d_buffer    = d_chars_ + d_offsets_[idx];

    int charOffsetToLastSpace = -1;
    int byteOffsetToLastSpace = -1;
    int spos                  = 0;
    int bidx                  = 0;

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const the_chr = *itr;
      auto const pos     = itr.position();

      // execute conditions:
      if (the_chr <= ' ') {  // convert all whitespace to space
        d_buffer[bidx]        = ' ';
        byteOffsetToLastSpace = bidx;
        charOffsetToLastSpace = pos;
      }
      if (pos - spos >= width_ && byteOffsetToLastSpace >= 0) {
        d_buffer[byteOffsetToLastSpace] = '\n';
        spos                            = charOffsetToLastSpace;
        byteOffsetToLastSpace           = -1;
        charOffsetToLastSpace           = -1;
      }
      bidx += detail::bytes_in_char_utf8(the_chr);
    }
    return 0;
  }

 private:
  column_device_view const d_column_;
  int32_t const* d_offsets_;
  char* d_chars_;
  size_type width_;
};

}  // namespace

template <typename device_execute_functor>
std::unique_ptr<column> wrap(
  strings_column_view const& strings,
  size_type width,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(width > 0, "Positive wrap width required");

  auto strings_count = strings.size();
  if (strings_count == 0) return detail::make_empty_strings_column(mr, stream);

  auto execpol = rmm::exec_policy(stream);

  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  size_type null_count = strings.null_count();

  // copy null mask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);

  // build offsets column
  auto offsets_column = std::make_unique<column>(strings.offsets(), stream, mr);  // makes a copy
  auto d_new_offsets  = offsets_column->view().template data<int32_t>();

  auto chars_column = std::make_unique<column>(strings.chars(), stream, mr);  // makes a copy
  auto d_chars      = chars_column->mutable_view().data<char>();

  device_execute_functor d_execute_fctr{d_column, d_new_offsets, d_chars, width};

  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     d_execute_fctr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

std::unique_ptr<column> wrap(strings_column_view const& strings,
                             size_type width,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::wrap<detail::execute_wrap>(strings, width, mr);
}

}  // namespace strings
}  // namespace cudf
