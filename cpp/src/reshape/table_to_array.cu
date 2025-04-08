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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace {

template <typename T>
void _table_to_device_array(cudf::table_view const& input,
                            void* output,
                            rmm::cuda_stream_view stream)
{
  auto const num_columns = input.num_columns();
  auto const num_rows    = input.num_rows();
  auto const item_size   = sizeof(T);

  std::vector<void*> dsts(num_columns);
  std::vector<void const*> srcs(num_columns);
  std::vector<size_t> sizes(num_columns, item_size * num_rows);

  auto* base_ptr = static_cast<uint8_t*>(output);

  for (int i = 0; i < num_columns; ++i) {
    auto const& col = input.column(i);
    CUDF_EXPECTS(col.type() == input.column(0).type(), "All columns must have the same dtype");

    auto* src_ptr = static_cast<void const*>(col.data<T>());
    auto* dst_ptr = base_ptr + i * item_size * num_rows;

    srcs[i] = src_ptr;
    dsts[i] = dst_ptr;
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  std::vector<cudaMemcpyAttributes> attrs{attr};
  std::vector<size_t> attr_idxs{0};
  size_t fail_idx = SIZE_MAX;

  CUDF_CUDA_TRY(cudaMemcpyBatchAsync(dsts.data(),
                                     const_cast<void**>(srcs.data()),
                                     sizes.data(),
                                     num_columns,
                                     attrs.data(),
                                     attr_idxs.data(),
                                     attrs.size(),
                                     &fail_idx,
                                     stream.value()));
#else
  for (int i = 0; i < num_columns; ++i) {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDeviceToDevice, stream.value()));
  }
#endif
}

struct TableToArrayDispatcher {
  table_view const& input;
  void* output;
  rmm::cuda_stream_view stream;

  template <typename T, CUDF_ENABLE_IF(is_fixed_width<T>() || is_fixed_point<T>())>
  void operator()() const
  {
    if constexpr (is_fixed_point<T>()) {
      using StorageType = cudf::device_storage_type_t<T>;
      _table_to_device_array<StorageType>(input, output, stream);
    } else {
      _table_to_device_array<T>(input, output, stream);
    }
  }

  template <typename T, CUDF_ENABLE_IF(!is_fixed_width<T>() && !is_fixed_point<T>())>
  void operator()() const
  {
    CUDF_FAIL("Unsupported dtype");
  }
};

}  // namespace

void table_to_device_array(table_view const& input,
                           void* output,
                           data_type output_dtype,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref)
{
  CUDF_FUNC_RANGE();
  cudf::type_dispatcher(output_dtype, TableToArrayDispatcher{input, output, stream});
}

}  // namespace cudf
