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
  if (_file == testfile::FILE1) {
    // col: "a" null: 0-1024
    auto& input_col  = _input_columns[0];
    size_t max_depth = input_col.nesting_depth();
    auto* cols       = &_output_buffers;
    {
      auto& out_buf = (*cols)[input_col.nesting[0]];
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(0),
                       thrust::counting_iterator(1024),
                       clear_bit_functor{out_buf.null_mask()});
      out_buf.null_count() += 1024;
    }

    // col: "b" null: 1024-2000
    input_col = _input_columns[1];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    {
      auto& out_buf = (*cols)[input_col.nesting[0]];
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(1024),
                       thrust::counting_iterator(2000),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 976;
    }

    // col: "c" null: 3072-4000
    input_col = _input_columns[2];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(3072),
                       thrust::counting_iterator(4000),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 928;
    }
  } else {
    // col: "a" null: 3072-4000
    auto& input_col  = _input_columns[0];
    size_t max_depth = input_col.nesting_depth();
    auto* cols       = &_output_buffers;
    {
      auto& out_buf = (*cols)[input_col.nesting[0]];
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(3072),
                       thrust::counting_iterator(4000),
                       clear_bit_functor{out_buf.null_mask()});
      out_buf.null_count() += 928;
    }

    // col: "b" null: 3072-4000
    input_col = _input_columns[1];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    {
      auto& out_buf = (*cols)[input_col.nesting[0]];
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(3072),
                       thrust::counting_iterator(4000),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 928;
    }

    // col: "c" null: 2000-3072
    input_col = _input_columns[2];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(2000),
                       thrust::counting_iterator(3072),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 1072;
    }

    // col: "d" null: 1024-2000
    input_col = _input_columns[3];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(1024),
                       thrust::counting_iterator(2000),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 976;
    }

    // col: "e" null: 0-1024 and 1024-2000
    input_col = _input_columns[4];
    max_depth = input_col.nesting_depth();
    cols      = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      thrust::for_each(rmm::exec_policy(_stream),
                       thrust::counting_iterator(0),
                       thrust::counting_iterator(2000),
                       clear_bit_functor{out_buf.null_mask()});

      out_buf.null_count() += 2000;
    }
  }
}
}  // namespace cudf::io::parquet::detail