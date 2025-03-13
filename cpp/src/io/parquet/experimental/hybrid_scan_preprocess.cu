/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/functional.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <bitset>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

namespace {

__device__ constexpr bool is_string_chunk(cudf::io::parquet::detail::ColumnChunkDesc const& chunk)
{
  auto const is_decimal =
    chunk.logical_type.has_value() and
    chunk.logical_type->type == cudf::io::parquet::detail::LogicalType::DECIMAL;
  auto const is_binary = chunk.physical_type == cudf::io::parquet::detail::BYTE_ARRAY or
                         chunk.physical_type == cudf::io::parquet::detail::FIXED_LEN_BYTE_ARRAY;
  return is_binary and not is_decimal;
}

struct set_str_dict_index_count {
  device_span<size_t> str_dict_index_count;
  device_span<cudf::io::parquet::detail::ColumnChunkDesc const> chunks;

  __device__ void operator()(cudf::io::parquet::detail::PageInfo const& page)
  {
    auto const& chunk = chunks[page.chunk_idx];
    if ((page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) != 0 and
        chunk.num_dict_pages > 0 and is_string_chunk(chunk)) {
      // there is only ever one dictionary page per chunk, so this is safe to do in parallel.
      str_dict_index_count[page.chunk_idx] = page.num_input_values;
    }
  }
};

struct set_str_dict_index_ptr {
  cudf::io::parquet::detail::string_index_pair* const base;
  device_span<size_t const> str_dict_index_offsets;
  device_span<cudf::io::parquet::detail::ColumnChunkDesc> chunks;

  __device__ void operator()(size_t i)
  {
    auto& chunk = chunks[i];
    if (chunk.num_dict_pages > 0 and is_string_chunk(chunk)) {
      chunk.str_dict_index = base + str_dict_index_offsets[i];
    }
  }
};

}  // namespace

void impl::prepare_row_groups(cudf::host_span<std::vector<size_type> const> row_group_indices,
                              cudf::io::parquet_reader_options const& options)
{
  std::tie(
    _file_itm_data.global_skip_rows, _file_itm_data.global_num_rows, _file_itm_data.row_groups) =
    _metadata->add_row_groups(row_group_indices, options.get_skip_rows(), options.get_num_rows());

  // check for page indexes
  _has_page_index = std::all_of(_file_itm_data.row_groups.cbegin(),
                                _file_itm_data.row_groups.cend(),
                                [](auto const& row_group) { return row_group.has_page_index(); });

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty()) {
    // fills in chunk information without physically loading or decompressing
    // the associated data
    create_global_chunk_info(options);

    // compute schedule of input reads.
    compute_input_passes();
  }

  _file_preprocessed = true;
}

void impl::allocate_level_decode_space()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto& pages = subpass.pages;

  // TODO: this could be made smaller if we ignored dictionary pages and pages with no
  // repetition data.
  size_t const per_page_decode_buf_size =
    cudf::io::parquet::detail::LEVEL_DECODE_BUF_SIZE * 2 * pass.level_type_size;
  auto const decode_buf_size = per_page_decode_buf_size * pages.size();
  subpass.level_decode_data =
    rmm::device_buffer(decode_buf_size, _stream, cudf::get_current_device_resource_ref());

  // distribute the buffers
  auto* buf = static_cast<uint8_t*>(subpass.level_decode_data.data());
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto& p = pages[idx];

    p.lvl_decode_buf[cudf::io::parquet::detail::level_type::DEFINITION] = buf;
    buf += (cudf::io::parquet::detail::LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
    p.lvl_decode_buf[cudf::io::parquet::detail::level_type::REPETITION] = buf;
    buf += (cudf::io::parquet::detail::LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
  }
}

void impl::build_string_dict_indices()
{
  CUDF_FUNC_RANGE();

  auto& pass = *_pass_itm_data;

  // compute number of indices per chunk and a summed total
  rmm::device_uvector<size_t> str_dict_index_count(pass.chunks.size() + 1, _stream);
  thrust::fill(
    rmm::exec_policy_nosync(_stream), str_dict_index_count.begin(), str_dict_index_count.end(), 0);
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   set_str_dict_index_count{str_dict_index_count, pass.chunks});

  size_t const total_str_dict_indexes = thrust::reduce(
    rmm::exec_policy(_stream), str_dict_index_count.begin(), str_dict_index_count.end());
  if (total_str_dict_indexes == 0) { return; }

  // convert to offsets
  rmm::device_uvector<size_t>& str_dict_index_offsets = str_dict_index_count;
  thrust::exclusive_scan(rmm::exec_policy_nosync(_stream),
                         str_dict_index_offsets.begin(),
                         str_dict_index_offsets.end(),
                         str_dict_index_offsets.begin(),
                         0);

  // allocate and distribute pointers
  pass.str_dict_index =
    cudf::detail::make_zeroed_device_uvector_async<cudf::io::parquet::detail::string_index_pair>(
      total_str_dict_indexes, _stream, cudf::get_current_device_resource_ref());

  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy_nosync(_stream),
    iter,
    iter + pass.chunks.size(),
    set_str_dict_index_ptr{pass.str_dict_index.data(), str_dict_index_offsets, pass.chunks});

  // compute the indices
  BuildStringDictionaryIndex(pass.chunks.device_ptr(), pass.chunks.size(), _stream);
  pass.chunks.device_to_host_sync(_stream);
}

}  // namespace cudf::experimental::io::parquet::detail
