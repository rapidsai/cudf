/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <strings/utilities.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
using translate_table = thrust::pair<char_utf8, char_utf8>;

namespace {
/**
 * @brief This is the translate functor for replacing individual characters
 * in each string.
 */
struct translate_fn {
  column_device_view const d_strings;
  rmm::device_uvector<translate_table>::iterator table_begin;
  rmm::device_uvector<translate_table>::iterator table_end;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    string_view const d_str = d_strings.element<string_view>(idx);

    size_type bytes = d_str.size_bytes();
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto chr : d_str) {
      auto const entry =
        thrust::lower_bound(thrust::seq,
                            table_begin,
                            table_end,
                            translate_table{chr, 0},
                            [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
      if (entry != table_end && entry->first == chr) {
        bytes -= bytes_in_char_utf8(chr);
        chr = entry->second;
        if (chr)  // if null, skip the character
          bytes += bytes_in_char_utf8(chr);
      }
      if (chr && out_ptr) out_ptr += from_char_utf8(chr, out_ptr);
    }
    if (!d_chars) d_offsets[idx] = bytes;
  }
};

}  // namespace

//
std::unique_ptr<column> translate(
  strings_column_view const& strings,
  std::vector<std::pair<char_utf8, char_utf8>> const& chars_table,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);

  size_type table_size = static_cast<size_type>(chars_table.size());
  // convert input table
  thrust::host_vector<translate_table> htable(table_size);
  std::transform(chars_table.begin(), chars_table.end(), htable.begin(), [](auto entry) {
    return translate_table{entry.first, entry.second};
  });
  // The size of this table is usually much less than 100 so it is was
  // found to be more efficient to sort on the CPU than the GPU.
  thrust::sort(htable.begin(), htable.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.first < rhs.first;
  });
  // copy translate table to device memory
  rmm::device_uvector<translate_table> table(htable.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(table.data(),
                           htable.data(),
                           sizeof(translate_table) * htable.size(),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  auto d_strings = column_device_view::create(strings.parent(), stream);

  auto children = make_strings_children(translate_fn{*d_strings, table.begin(), table.end()},
                                        strings.size(),
                                        strings.null_count(),
                                        stream,
                                        mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> translate(strings_column_view const& strings,
                                  std::vector<std::pair<uint32_t, uint32_t>> const& chars_table,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::translate(strings, chars_table, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
