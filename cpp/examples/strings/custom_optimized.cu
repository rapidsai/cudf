/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "common.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

/**
 * @brief Computes the size of each output row
 *
 * This thread is called once per row in d_names.
 *
 * @param d_names Column of names
 * @param d_visibilities Column of visibilities
 * @param d_sizes Output sizes for each row
 */
__global__ void sizes_kernel(cudf::column_device_view const d_names,
                             cudf::column_device_view const d_visibilities,
                             cudf::size_type* d_sizes)
{
  // The row index is resolved from the CUDA thread/block objects
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  // There may be more threads than actual rows
  if (index >= d_names.size()) return;

  auto const visible   = cudf::string_view("public", 6);
  auto const redaction = cudf::string_view("X X", 3);

  auto const name = d_names.element<cudf::string_view>(index);
  auto const vis  = d_visibilities.element<cudf::string_view>(index);

  cudf::size_type result = redaction.size_bytes();  // init to redaction size
  if (vis == visible) {
    auto const space_idx    = name.find(' ');
    auto const first        = name.substr(0, space_idx);
    auto const last_initial = name.substr(space_idx + 1, 1);

    result = first.size_bytes() + last_initial.size_bytes() + 1;
  }

  d_sizes[index] = result;
}

/**
 * @brief Builds the output for each row
 *
 * This thread is called once per row in d_names.
 *
 * @param d_names Column of names
 * @param d_visibilities Column of visibilities
 * @param d_offsets Byte offset in `d_chars` for each row
 * @param d_chars Output memory for all rows
 */
__global__ void redact_kernel(cudf::column_device_view const d_names,
                              cudf::column_device_view const d_visibilities,
                              cudf::size_type const* d_offsets,
                              char* d_chars)
{
  // The row index is resolved from the CUDA thread/block objects
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  // There may be more threads than actual rows
  if (index >= d_names.size()) return;

  auto const visible   = cudf::string_view("public", 6);
  auto const redaction = cudf::string_view("X X", 3);

  // resolve output_ptr using the offsets vector
  char* output_ptr = d_chars + d_offsets[index];

  auto const name = d_names.element<cudf::string_view>(index);
  auto const vis  = d_visibilities.element<cudf::string_view>(index);

  if (vis == visible) {
    auto const space_idx    = name.find(' ');
    auto const first        = name.substr(0, space_idx);
    auto const last_initial = name.substr(space_idx + 1, 1);
    auto const output_size  = first.size_bytes() + last_initial.size_bytes() + 1;

    // build output string
    memcpy(output_ptr, last_initial.data(), last_initial.size_bytes());
    output_ptr += last_initial.size_bytes();
    *output_ptr++ = ' ';
    memcpy(output_ptr, first.data(), first.size_bytes());
  } else {
    memcpy(output_ptr, redaction.data(), redaction.size_bytes());
  }
}

/**
 * @brief Redacts each name per the corresponding visibility entry
 *
 * This implementation builds the strings column children (offsets and chars)
 * directly into device memory for libcudf.
 *
 * @param names Column of names
 * @param visibilities Column of visibilities
 * @return Redacted column of names
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities)
{
  // all device memory operations and kernel functions will run on this stream
  auto stream = rmm::cuda_stream_default;

  auto const d_names        = cudf::column_device_view::create(names, stream);
  auto const d_visibilities = cudf::column_device_view::create(visibilities, stream);

  constexpr int block_size = 128;  // this arbitrary size should be a power of 2
  int const blocks         = (names.size() + block_size - 1) / block_size;

  nvtxRangePushA("redact_strings");

  // create offsets vector
  auto offsets = rmm::device_uvector<cudf::size_type>(names.size() + 1, stream);

  // compute output sizes
  sizes_kernel<<<blocks, block_size, 0, stream.value()>>>(
    *d_names, *d_visibilities, offsets.data());

  // convert sizes to offsets (in place)
  thrust::exclusive_scan(rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());

  // last element is the total output size
  // (device-to-host copy of 1 integer -- includes syncing the stream)
  cudf::size_type output_size = offsets.back_element(stream);

  //  create chars vector
  auto chars = rmm::device_uvector<char>(output_size, stream);

  // build chars output
  redact_kernel<<<blocks, block_size, 0, stream.value()>>>(
    *d_names, *d_visibilities, offsets.data(), chars.data());

  // create column from offsets and chars vectors (no copy is performed)
  auto result = cudf::make_strings_column(names.size(), std::move(offsets), std::move(chars));

  // wait for all of the above to finish
  stream.synchronize();

  nvtxRangePop();
  return result;
}
