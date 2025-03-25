/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "reader_impl.hpp"
#include "reader_impl_page_pruning.hpp"

#include <cudf/detail/utilities/functional.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

namespace cudf::io::parquet::detail {
namespace {

struct clear_bit_functor {
  cudf::bitmask_type* null_mask;
  __device__ void operator()(cudf::size_type idx) { cudf::clear_bit(null_mask, idx); }
};
}  // namespace

void reader::impl::fix_holes()
{
  using byte_range_info = cudf::io::text::byte_range_info;

  auto const update_null_count = [&](input_column_info& input_col,
                                     std::vector<byte_range_info> const& row_ranges) {
    size_t max_depth = input_col.nesting_depth();
    auto* cols       = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      for (auto const& row_range : row_ranges) {
        thrust::for_each(rmm::exec_policy(_stream),
                         thrust::counting_iterator(row_range.offset()),
                         thrust::counting_iterator(row_range.offset() + row_range.size()),
                         clear_bit_functor{out_buf.null_mask()});
        out_buf.null_count() += row_range.size();
      }
    }
  };

  switch (_file) {
    case testfile::FILE1: {
      // col: "a" null: 0-1024
      update_null_count(_input_columns[0], {byte_range_info(0, 1024)});
      // col: "b" null: 1024-2000
      update_null_count(_input_columns[1], {byte_range_info(1024, 976)});
      // col: "c" null: 3072-4000
      update_null_count(_input_columns[2], {byte_range_info(3072, 928)});
      break;
    }
    case testfile::FILE2: {
      // col: "a" null: 3072-4000
      update_null_count(_input_columns[0], {byte_range_info(3072, 928)});
      // col: "b" null: 3072-4000
      update_null_count(_input_columns[1], {byte_range_info(3072, 928)});
      // col: "c" null: 2000-3072
      update_null_count(_input_columns[2], {byte_range_info(2000, 1072)});
      // col: "d" null: 1024-2000
      update_null_count(_input_columns[3], {byte_range_info(1024, 976)});
      // col: "e" null: 0-1024 and 1024-2000
      update_null_count(_input_columns[4], {byte_range_info(0, 1024), byte_range_info(1024, 976)});
      break;
    }
    case testfile::FILE3: {
      // col: "a" null: 1024-2000
      update_null_count(_input_columns[0], {byte_range_info(1024, 976)});
      // col: "b" null: 3072-5120
      update_null_count(_input_columns[1], {byte_range_info(3072, 2048)});
      // col: "c" null: 1024-3072, 4000-5120
      update_null_count(_input_columns[2],
                        {byte_range_info(1024, 2048), byte_range_info(4000, 1120)});
      break;
    }
    case testfile::FILE4: {
      // col: "a" null: 1024-2000
      update_null_count(_input_columns[0], {byte_range_info(1024, 976)});
      // col: "b" null: 0-1024
      update_null_count(_input_columns[1], {byte_range_info(0, 1024)});
      // col: "c" null: 1024-2000
      update_null_count(_input_columns[2], {byte_range_info(1024, 976)});

      break;
    }
    case testfile::FILE5: {
      // col: "a" null: 1024-2048
      update_null_count(_input_columns[0], {byte_range_info(1024, 1024)});
      // col: "b" null: 2048-4000
      update_null_count(_input_columns[1], {byte_range_info(2048, 1952)});
      // col: "c" null: 0-4000
      update_null_count(_input_columns[2], {byte_range_info(0, 4000)});
      // col: "d" null: 3072-4000
      update_null_count(_input_columns[3], {byte_range_info(3072, 928)});
      // col: "e" null: 2048-4000
      update_null_count(_input_columns[4], {byte_range_info(2048, 1952)});
      // col: "f" null: 0-256, 1536-1792, 3999-4000
      update_null_count(
        _input_columns[5],
        {byte_range_info(0, 256), byte_range_info(1536, 256), byte_range_info(3999, 1)});

      // col: "g" null: 512-1024, 3072-3584, 3999-4000
      update_null_count(
        _input_columns[6],
        {byte_range_info(512, 512), byte_range_info(3072, 512), byte_range_info(3999, 1)});

      break;
    }
    case testfile::FILE6: {
      // col: "a" null: 1024-2000
      update_null_count(_input_columns[0], {byte_range_info(1024, 976)});
      // col: "b" null: 0-1024
      update_null_count(_input_columns[1], {byte_range_info(0, 1024)});
      // col: "c" null: 0-512, 1024-1536, 1999-2000
      update_null_count(
        _input_columns[2],
        {byte_range_info(0, 512), byte_range_info(1024, 512), byte_range_info(1999, 1)});
      // col: "d" null: 1024-1536, 1999-2000
      update_null_count(_input_columns[3], {byte_range_info(1024, 512), byte_range_info(1999, 1)});
      // col: "e" null: 0-2000
      update_null_count(_input_columns[4], {byte_range_info(0, 2000)});
      // col: "f" null: 1024-2000
      update_null_count(_input_columns[5], {byte_range_info(1024, 976)});
      break;
    }
    default: {
      break;
    }
  }
}

}  // namespace cudf::io::parquet::detail
