/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Convert tokens and row2tensor map to final tensor data.
 *
 * @param[in] token_ids Tokens from tokenizer
 * @param[in] offsets Offsets to each string's output row of tokens
 * @param[in] row2tensor String to tensor token counts
 * @param[in] row2row_within_tensor Token counts within sub-rows of the output
 * @param[in] max_sequence_length Maximum number of tokens in a row
 * @param[in] nrows_tensor_token_ids Total number of output tensor rows
 * @param[in] stride Number of tokens in sub-rows
 * @param[in] do_truncate True if tokens should not spill into sub-rows in the output
 * @param[out] final_tensor Output vector of token-ids
 * @param[out] attn_mask Identifies valid token id entries
 * @param[out] metadata Additional data per row
 */
template <typename InputType, typename OffsetType>
CUDF_KERNEL void kernel_compute_tensor_metadata(
  // input
  InputType const* token_ids,
  OffsetType const* offsets,
  InputType const* row2tensor,
  InputType const* row2row_within_tensor,
  InputType max_sequence_length,
  int64_t nrows_tensor_token_ids,
  InputType stride,
  bool do_truncate,
  // output
  uint32_t* final_tensor,
  uint32_t* attn_mask,
  uint32_t* metadata)
{
  auto const output_idx = cudf::detail::grid_1d::global_thread_id();

  auto const absolute_row_id = output_idx / max_sequence_length;
  if (absolute_row_id >= nrows_tensor_token_ids) { return; }
  auto const tensor_id               = row2tensor[absolute_row_id];
  auto const row_within_tensor       = row2row_within_tensor[absolute_row_id];
  auto const offset_token_ids_tensor = offsets[tensor_id];
  auto const n_tokens_tensor         = offsets[tensor_id + 1] - offset_token_ids_tensor;
  // check for last row within tensor
  bool const last_row_of_tensor = (absolute_row_id == nrows_tensor_token_ids - 1) ||
                                  (row2tensor[absolute_row_id + 1] != tensor_id);
  // compute input offset to retrieve token ids
  auto const token_idx = output_idx % max_sequence_length;
  auto const row_offset_token_ids =
    offset_token_ids_tensor + token_idx +
    (row_within_tensor ? (max_sequence_length + (stride * (row_within_tensor - 1))) : 0);

  if (row_within_tensor == 0) {
    if (token_idx < n_tokens_tensor) {
      // copy token ids
      final_tensor[output_idx] = token_ids[row_offset_token_ids];
      attn_mask[output_idx]    = 1;
    } else {
      // pad with 0
      final_tensor[output_idx] = 0;
      attn_mask[output_idx]    = 0;
    }
  } else {
    auto const n_replicates = max_sequence_length - stride;
    if ((row_offset_token_ids - n_replicates) < (offset_token_ids_tensor + n_tokens_tensor)) {
      // replicate elements from previous row or copy new tokens
      final_tensor[output_idx] = token_ids[row_offset_token_ids - n_replicates];
      attn_mask[output_idx]    = 1;
    } else {
      // pad with 0
      final_tensor[output_idx] = 0;
      attn_mask[output_idx]    = 0;
    }
  }

  // write metadata
  if (token_idx == 0) {
    auto const metadata_idx    = absolute_row_id * 3;  // three metadata values per output row
    metadata[metadata_idx]     = tensor_id;
    metadata[metadata_idx + 1] = (row_within_tensor == 0) ? 0 : (max_sequence_length - stride) / 2;
    metadata[metadata_idx + 2] = [&] {
      if (!last_row_of_tensor) {
        return static_cast<uint32_t>(max_sequence_length - (max_sequence_length - stride) / 2 - 1);
      }
      if (n_tokens_tensor <= max_sequence_length) {
        // we fit, all good
        return (n_tokens_tensor > 0) ? static_cast<uint32_t>(n_tokens_tensor - 1) : 0;
      }
      if (do_truncate) { return static_cast<uint32_t>(max_sequence_length - 1); }

      auto const final_row_value =
        (max_sequence_length - stride) + (n_tokens_tensor - max_sequence_length) % stride;
      return (final_row_value > 0) ? static_cast<uint32_t>(final_row_value - 1) : 0;
    }();
  }
}

// this happens if there are no tokens in the input
tokenizer_result build_empty_result(cudf::size_type size,
                                    uint32_t max_sequence_length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto zero = cudf::numeric_scalar<uint32_t>(0, true, stream);
  auto ids  = cudf::detail::sequence(size * max_sequence_length, zero, zero, stream, mr);
  auto mask = cudf::detail::sequence(size * max_sequence_length, zero, zero, stream, mr);

  auto metadata = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, size * 3, cudf::mask_state::UNALLOCATED, stream, mr);
  thrust::tabulate(rmm::exec_policy(stream),
                   metadata->mutable_view().begin<uint32_t>(),
                   metadata->mutable_view().end<uint32_t>(),
                   [] __device__(auto idx) { return ((idx % 3) == 0) ? idx : 0; });
  metadata->set_null_count(0);

  return tokenizer_result{
    0, max_sequence_length, std::move(ids), std::move(mask), std::move(metadata)};
}

}  // namespace

tokenizer_result tokenized_to_tensor(cudf::lists_column_view const& input,
                                     cudf::size_type max_sequence_length,
                                     cudf::size_type stride,
                                     bool do_truncate,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(stride <= max_sequence_length,
               "stride must be less than or equal to max_sequence_length",
               std::invalid_argument);
  if (input.size() == input.null_count()) {  // empty or all-null returns empty
    auto empty_col = cudf::make_empty_column(cudf::type_id::UINT32);
    auto ulen      = static_cast<uint32_t>(max_sequence_length);
    return tokenizer_result{
      0, ulen, std::move(empty_col), std::move(empty_col), std::move(empty_col)};
  }
  CUDF_EXPECTS(max_sequence_length <= (std::numeric_limits<cudf::size_type>::max() / input.size()),
               "max_sequence_length times number of input rows exceeds the column size limit",
               std::overflow_error);

  auto token_ids       = cudf::column_device_view::create(input.child(), stream);
  auto d_token_ids     = token_ids->data<cudf::size_type>();
  auto token_offsets   = cudf::column_device_view::create(input.offsets(), stream);
  auto d_token_offsets = token_offsets->data<cudf::size_type>();

  // compute tensor offsets using existing offsets and accounting for the
  // stride, max_sequence_length, and do_truncate
  auto offsets_per_tensor   = rmm::device_uvector<cudf::size_type>(input.size() + 1, stream);
  auto d_offsets_per_tensor = offsets_per_tensor.data();
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(input.size()),
    d_offsets_per_tensor,
    cuda::proclaim_return_type<cudf::size_type>(
      [d_token_offsets, do_truncate, max_sequence_length, stride] __device__(auto idx) {
        auto const num_tokens = d_token_offsets[idx + 1] - d_token_offsets[idx];
        if (do_truncate || num_tokens <= max_sequence_length) return cudf::size_type{1};
        return 1 + ((num_tokens - max_sequence_length + stride - 1) / stride);
      }));

  auto nrows_tensor_token_ids = cudf::detail::sizes_to_offsets(
    offsets_per_tensor.begin(), offsets_per_tensor.end(), offsets_per_tensor.begin(), 0, stream);

  // if there are no tokens at all, build a specific empty result
  if (nrows_tensor_token_ids == 0) {
    return build_empty_result(input.size(), max_sequence_length, stream, mr);
  }

  // label row to tensor and row within tensor row
  auto row2tensor            = rmm::device_uvector<cudf::size_type>(nrows_tensor_token_ids, stream);
  auto d_row2tensor          = row2tensor.data();
  auto row2row_within_tensor = rmm::device_uvector<cudf::size_type>(nrows_tensor_token_ids, stream);
  auto d_row2row_within_tensor = row2row_within_tensor.data();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    input.size(),
    [d_offsets_per_tensor, d_row2tensor, d_row2row_within_tensor] __device__(auto idx) {
      auto const offset = d_offsets_per_tensor[idx];
      auto const size   = d_offsets_per_tensor[idx + 1] - offset;
      for (auto jdx = 0; jdx < size; ++jdx) {
        d_row2tensor[jdx + offset]            = idx;
        d_row2row_within_tensor[jdx + offset] = jdx;
      }
    });

  // create output data columns
  auto tensor_token_ids = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                                                    nrows_tensor_token_ids * max_sequence_length,
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  auto tensor_attention_mask =
    cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                              nrows_tensor_token_ids * max_sequence_length,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto tensor_metadata = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
                                                   nrows_tensor_token_ids * 3,
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);

  // compute final tensor, mask, and metadata
  constexpr int block_size = 256;
  cudf::detail::grid_1d const grid{
    static_cast<cudf::size_type>(nrows_tensor_token_ids * max_sequence_length), block_size};
  kernel_compute_tensor_metadata<<<grid.num_blocks,
                                   grid.num_threads_per_block,
                                   0,
                                   stream.value()>>>(
    d_token_ids,
    d_token_offsets,
    d_row2tensor,
    d_row2row_within_tensor,
    max_sequence_length,
    nrows_tensor_token_ids,
    stride,
    do_truncate,
    tensor_token_ids->mutable_view().data<uint32_t>(),
    tensor_attention_mask->mutable_view().data<uint32_t>(),
    tensor_metadata->mutable_view().data<uint32_t>());

  return tokenizer_result{static_cast<uint32_t>(nrows_tensor_token_ids),
                          static_cast<uint32_t>(max_sequence_length),
                          std::move(tensor_token_ids),
                          std::move(tensor_attention_mask),
                          std::move(tensor_metadata)};
}

}  // namespace detail

tokenizer_result tokenized_to_tensor(cudf::lists_column_view const& input,
                                     cudf::size_type max_sequence_length,
                                     cudf::size_type stride,
                                     bool do_truncate,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenized_to_tensor(input, max_sequence_length, stride, do_truncate, stream, mr);
}

}  // namespace nvtext
