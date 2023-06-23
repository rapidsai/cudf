/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <io/parquet/decode_fixed.hpp>
#include <io/parquet/decode_general.hpp>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

/**
 * @copydoc cudf::io::parquet::gpu::DecodePageData
 */
void __host__ DecodePageData(cudf::detail::hostdevice_vector<PageInfo>& pages,
                             cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There are no pages to decode");  
  
  // determine which kernels to invoke
  /*
  auto mask_iter = thrust::make_transform_iterator(
    pages.begin(), [] __device__(PageInfo const& p) { return p.kernel_mask; });
  int kernel_mask = thrust::reduce(
    rmm::exec_policy(stream), mask_iter, mask_iter + pages.size(), 0, thrust::bit_or<int>{});
    */

  // invoke all relevant kernels. each one will only process the pages whose masks match
  // their own, and early-out on the rest.
  //if (kernel_mask & KERNEL_MASK_FIXED_WIDTH_NO_DICT) {
//    DecodePageDataFixed(pages, chunks, num_rows, min_row, level_type_size, stream);
//  }
//  if (kernel_mask & KERNEL_MASK_GENERAL) {
    DecodePageDataGeneral(pages, chunks, num_rows, min_row, level_type_size, stream);
//  }
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
