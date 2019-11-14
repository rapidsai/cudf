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
#include <cudf/transpose.hpp>
#include <cudf/detail/transpose.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>

namespace cudf {

namespace {

using experimental::detail::warp_size;
constexpr int MAX_GRID_SIZE = (1<<16)-1;

/**
 * @brief Transposes the values from ncols x nrows input to nrows x ncols output 
 * 
 * @tparam T  Datatype of values pointed to by the views
 * @param input[in]  Device view of input columns' data
 * @param output[out]  Mutable device view of pre-allocated output columns' data
 */
template <typename T>
__global__
void gpu_transpose(table_device_view const input, mutable_table_device_view output)
{
  size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  size_type stride_x = blockDim.x * gridDim.x;
  size_type stride_y = blockDim.y * gridDim.y;

  for (size_type i = x; i < input.num_columns(); i += stride_x) {
    for (size_type j = y; j < input.num_rows(); j += stride_y) {
      output.column(j).element<T>(i) = input.column(i).element<T>(j);
    }
  }
}

/**
 * @brief Transposes the null mask from ncols x nrows input to nrows x ncols output
 * 
 * @tparam T  Datatype of values pointed to by the views
 * @param input[in]  Device view of input columns' data
 * @param output[out]  Mutable device view of pre-allocated output columns' data
 */
__global__
void gpu_transpose_valids(table_device_view const input, mutable_table_device_view output,
                          size_type *null_count)
{
  size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  size_type stride_x = blockDim.x * gridDim.x;
  size_type stride_y = blockDim.y * gridDim.y;

  size_type i = x;
  auto active_threads = __ballot_sync(0xffffffff, i < input.num_columns());
  while (i < input.num_columns()) {
    for (size_type j = y; j < input.num_rows(); j += stride_y) {
      auto const result = __ballot_sync(active_threads, input.column(i).is_valid(j));

      // Only one thread writes output
      if (0 == threadIdx.x % warp_size) {
        output.column(j).set_mask_word(word_index(i), result);
        int num_nulls = __popc(active_threads) - __popc(result);
        atomicAdd(null_count + j, num_nulls);
      }
    }

    // Update active threads before branching
    i += stride_x;
    active_threads = __ballot_sync(active_threads, i < input.num_columns());
  }
}

struct launch_kernel {
  template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<experimental::table> operator()(table_view const& input,
      bool allocate_nulls, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // Output has transposed shape of input
    auto const output_ncols = input.num_rows();
    auto const output_nrows = input.num_columns();

    // If any input column has nulls, all output columns must
    auto const& first = input.column(0);
    auto const allocation_policy = allocate_nulls
      ? experimental::mask_allocation_policy::ALWAYS
      : experimental::mask_allocation_policy::NEVER;

    // Allocate output columns
    std::vector<std::unique_ptr<column>> output(output_ncols);
    std::generate(output.begin(), output.end(), [=, &first]() {
      return experimental::detail::allocate_like(first, output_nrows,
        allocation_policy, mr, stream);
    });

    // Create mutable device view from input
    auto device_input = table_device_view::create(input, stream);

    // Create mutable table device view from output columns
    auto output_views = std::vector<mutable_column_view>(output.size());
    std::transform(output.begin(), output.end(), output_views.begin(),
      [](auto const& col) { return static_cast<mutable_column_view>(*col); });
    auto output_table = mutable_table_view(output_views);
    auto device_output = mutable_table_device_view::create(output_table, stream);

    // TODO benchmark, because a smaller block size (e.g. 32x8) may perform
    // better. It would also require transposing via shared memory, which may
    // improve performance anyway because it would enable coalesced loads and
    // stores rather than one or the other.
    dim3 dimBlock(warp_size, warp_size);
    dim3 dimGrid(std::min(util::div_rounding_up_safe(input.num_columns(), warp_size), MAX_GRID_SIZE),
                 std::min(util::div_rounding_up_safe(input.num_rows(), warp_size), MAX_GRID_SIZE));

    gpu_transpose<T><<<dimGrid, dimBlock, 0, stream>>>(*device_input, *device_output);

    if (allocate_nulls) {
      // Transpose valids and compute null counts
      rmm::device_vector<size_type> d_null_counts(output.size());
      gpu_transpose_valids<<<dimGrid, dimBlock, 0, stream>>>(*device_input,
        *device_output, d_null_counts.data().get());

      // Set null counts on output columns
      thrust::host_vector<size_type> null_counts(d_null_counts);
      auto begin = thrust::make_zip_iterator(thrust::make_tuple(output.begin(), null_counts.begin()));
      auto end = thrust::make_zip_iterator(thrust::make_tuple(output.end(), null_counts.end()));
      thrust::for_each(thrust::host, begin, end,
        [](thrust::tuple<std::unique_ptr<column>&, size_type const&> tuple) {
          auto& out_col = thrust::get<0>(tuple);
          auto const& null_count = thrust::get<1>(tuple);
          out_col->set_null_count(null_count);
        });;
    }

    return std::make_unique<experimental::table>(std::move(output));
  }

  template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<experimental::table> operator()(table_view const& input,
      bool allocate_nulls, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // TODO add string support
    CUDF_FAIL("Invalid, non-fixed-width type");
  }
};

}  // namespace

namespace detail {

std::unique_ptr<experimental::table> transpose(table_view const& input,
  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // If there are no rows in the input, return successfully
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return std::make_unique<experimental::table>();
  }

  // Check datatype homogeneity
  auto const dtype = input.column(0).type();
  CUDF_EXPECTS(std::all_of(input.begin(), input.end(), [dtype](auto const& col) {
    return dtype == col.type(); }), "Column type mismatch");

  nvtx::range_push("CUDF_TRANSPOSE", nvtx::color::GREEN);
  auto output = experimental::type_dispatcher(dtype, launch_kernel{}, input,
    has_nulls(input), mr, stream);
  nvtx::range_pop();

  return output;
}

}  // namespace detail

std::unique_ptr<experimental::table> transpose(table_view const& input,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::transpose(input, mr);
}

}  // namespace cudf
