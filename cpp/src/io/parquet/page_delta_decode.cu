/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "delta_binary.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/exec_policy.hpp>
#include <thrust/transform_scan.h>

namespace cudf::io::parquet::gpu {

namespace {

// Decode page data that is DELTA_BINARY_PACKED encoded. This encoding is
// only used for int32 and int64 physical types (and appears to only be used
// with V2 page headers; see https://www.mail-archive.com/dev@parquet.apache.org/msg11826.html).
// this kernel only needs 96 threads (3 warps)(for now).
template <typename level_t>
__global__ void __launch_bounds__(96)
  gpuDecodeDeltaBinary(PageInfo* pages,
                       device_span<ColumnChunkDesc const> chunks,
                       size_t min_row,
                       size_t num_rows,
                       int32_t* error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 0, 0> state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  int const lane_id     = t % warp_size;
  auto* const db        = &db_state;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{KERNEL_MASK_DELTA_BINARY},
                          true)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[delta_rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[delta_rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->page.skipped_leaf_values;

  // initialize delta state
  if (t == 0) { db->init_binary_block(s->data_start, s->data_end); }
  __syncthreads();

  auto const batch_size = db->values_per_mb;

  // if skipped_leaf_values is non-zero, then we need to decode up to the first mini-block
  // that has a value we need.
  if (skipped_leaf_values > 0) { db->skip_values(skipped_leaf_values); }

  while (s->error == 0 &&
         (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (t < 2 * warp_size) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->nz_count, src_pos + batch_size);
    }
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs values
    // into the proper location in the output.
    if (t < warp_size) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < 2 * warp_size) {
      // warp 1
      db->decode_batch();

    } else if (src_pos < target_pos) {
      // warp 2
      // nesting level that is storing actual leaf values
      int const leaf_level_index = s->col.max_nesting_depth - 1;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + lane_id; sp < src_pos + batch_size; sp += 32) {
        // the position in the output column/buffer
        int32_t dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->first_row; }

        // place value for this thread
        if (dst_pos >= 0 && sp < target_pos) {
          void* const dst = nesting_info_base[leaf_level_index].data_out + dst_pos * s->dtype_len;
          switch (s->dtype_len) {
            case 1:
              *static_cast<int8_t*>(dst) =
                db->value[rolling_index<delta_rolling_buf_size>(sp + skipped_leaf_values)];
              break;
            case 2:
              *static_cast<int16_t*>(dst) =
                db->value[rolling_index<delta_rolling_buf_size>(sp + skipped_leaf_values)];
              break;
            case 4:
              *static_cast<int32_t*>(dst) =
                db->value[rolling_index<delta_rolling_buf_size>(sp + skipped_leaf_values)];
              break;
            case 8:
              *static_cast<int64_t*>(dst) =
                db->value[rolling_index<delta_rolling_buf_size>(sp + skipped_leaf_values)];
              break;
          }
        }
      }

      if (lane_id == 0) { s->src_pos = src_pos + batch_size; }
    }
    __syncthreads();
  }

  if (t == 0 and s->error != 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device> ref{*error_code};
    ref.fetch_or(s->error, cuda::std::memory_order_relaxed);
  }
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::gpu::DecodeDeltaBinary
 */
void __host__ DecodeDeltaBinary(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                                size_t num_rows,
                                size_t min_row,
                                int level_type_size,
                                int32_t* error_code,
                                rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(96, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeDeltaBinary<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeDeltaBinary<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

}  // namespace cudf::io::parquet::gpu
