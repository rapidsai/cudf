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

#include "copy_if.cuh"
#include <cudf/legacy/table.hpp>
#include <thrust/logical.h>
#include <thrust/count.h>
 
namespace {

using bit_mask_t = bit_mask::bit_mask_t;

// Returns true if the valid mask is true for index i in at least keep_threshold
// columns
struct valid_table_filter
{
  __device__ inline 
  bool operator()(cudf::size_type i)
  {
    auto valid = [i](auto mask) { 
      return (mask == nullptr) || bit_mask::is_valid(mask, i);
    };

    auto count =
      thrust::count_if(thrust::seq, d_masks, d_masks + num_columns, valid);

    return (count >= keep_threshold);
  }

  static auto create(cudf::table const &table,
                     cudf::size_type keep_threshold,
                     cudaStream_t stream = 0)
  {
    std::vector<bit_mask_t*> h_masks(table.num_columns());

    std::transform(std::cbegin(table), std::cend(table), std::begin(h_masks),
      [](auto col) { return reinterpret_cast<bit_mask_t*>(col->valid); }
    );    
    
    size_t masks_size = sizeof(bit_mask_t*) * table.num_columns();

    bit_mask_t **device_masks = nullptr;
    RMM_TRY(RMM_ALLOC(&device_masks, masks_size, stream));
    CUDA_TRY(cudaMemcpyAsync(device_masks, h_masks.data(), masks_size,
                             cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(stream);

    auto deleter = [stream](valid_table_filter* f) { f->destroy(stream); };
    std::unique_ptr<valid_table_filter, decltype(deleter)> p {
      new valid_table_filter(device_masks, table.num_columns(), keep_threshold),
      deleter
    };

    CHECK_CUDA(stream);

    return p;
  }

  __host__ void destroy(cudaStream_t stream = 0) {
    RMM_FREE(d_masks, stream);
    delete this;
  }

  valid_table_filter() = delete;
  ~valid_table_filter() = default;

protected:

  valid_table_filter(bit_mask_t **masks,
                     cudf::size_type num_columns,
                     cudf::size_type keep_threshold) 
  : keep_threshold(keep_threshold),
    num_columns(num_columns),
    d_masks(masks) {}

  cudf::size_type keep_threshold;
  cudf::size_type num_columns;
  bit_mask_t **d_masks;
};

}  // namespace

namespace cudf {

/*
 * Filters a table to remove null elements.
 */
table drop_nulls(table const &input,
                 table const &keys,
                 cudf::size_type keep_threshold) {
  if (keys.num_columns() == 0 || keys.num_rows() == 0 ||
      not cudf::has_nulls(keys))
    return cudf::copy(input);

  CUDF_EXPECTS(keys.num_rows() <= input.num_rows(), 
               "Column size mismatch");

  auto filter = valid_table_filter::create(keys, keep_threshold);

  return detail::copy_if(input, *filter.get());
}

/*
 * Filters a table to remove null elements.
 */
table drop_nulls(table const &input,
                 table const &keys)
{
  return drop_nulls(input, keys, keys.num_columns());
}

}  // namespace cudf
