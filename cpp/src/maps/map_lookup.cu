/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf {
namespace {

void __device__ search_each_list(size_type list_index,
                        column_device_view input,
                        mutable_column_device_view output,
                        string_scalar_device_view lookup_key)
{
  if (input.is_null(list_index)) { // List row is null.
    output.element<size_type>(list_index) = -1;  // Not found.
    return;
  }

  auto offsets{input.child(0)};
  auto start_index{offsets.element<size_type>(list_index)};
  auto end_index{offsets.element<size_type>(list_index + 1)};

  auto key_column{input.child(1).child(0)};

  for (size_type list_element_index{start_index}; list_element_index < end_index; ++list_element_index) {
    if (!key_column.is_null(list_element_index) && key_column.element<string_view>(list_element_index) == lookup_key.value()) {
      output.element<size_type>(list_index) = list_element_index;
      return;
    }
  }

  output.element<size_type>(list_index) = -1;  // Not found.
}

template <int block_size>
__launch_bounds__(block_size) __global__
  void gpu_find_first(column_device_view input,
                      mutable_column_device_view output,
                      string_scalar_device_view lookup_key,
                      size_type* __restrict__ output_valid_count)
{
  size_type i      = blockIdx.x * block_size + threadIdx.x;
  size_type stride = block_size * gridDim.x;
  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < input.size());

  while (i < input.size()) {
    volatile bool output_valid{true};
    search_each_list(i, input, output, lookup_key);

    // Set validity mask.
    cudf::bitmask_type result_mask{__ballot_sync(active_threads, output_valid)};

    // Write out the mask.
    if (0 == threadIdx.x % cudf::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.size());
  }

  // Sum up valid_count across the whole block
  size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

size_type kernel_launcher(column_view const& input,
                          mutable_column_view const& output,
                          string_scalar& lookup_key,
                          cudaStream_t stream)
{
  constexpr size_type block_size{256};
  cudf::detail::grid_1d grid{input.size(), block_size};

  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto output_view       = mutable_column_device_view::create(output, stream);
  auto lookup_key_device_view{get_scalar_device_view(lookup_key)};
  rmm::device_scalar<size_type> device_valid_count{0, stream};

  gpu_find_first<block_size><<<grid.num_blocks, block_size, 0, stream>>>(
    *input_device_view, *output_view, lookup_key_device_view, device_valid_count.data());

  size_type valid_count = device_valid_count.value(stream);

  CHECK_CUDA(stream);

  return valid_count;
}
}  // namespace

std::unique_ptr<column> map_lookup(column_view const& map_column,
                                   string_scalar lookup_key,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream)
{
  // Defensive checks.
  CUDF_EXPECTS(map_column.type().id() == type_id::LIST, "Expected LIST<STRUCT<key,value>>.");

  lists_column_view lcv{map_column};
  auto structs_column = lcv.get_sliced_child(stream);

  CUDF_EXPECTS(structs_column.type().id() == type_id::STRUCT, "Expected LIST<STRUCT<key,value>>.");

  structs_column_view scv{structs_column};
  CUDF_EXPECTS(structs_column.num_children() == 2, "Expected LIST<STRUCT<key,value>>.");
  CUDF_EXPECTS(structs_column.child(0).type().id() == type_id::STRING,
               "Expected LIST<STRUCT<key,value>>.");
  CUDF_EXPECTS(structs_column.child(1).type().id() == type_id::STRING,
               "Expected LIST<STRUCT<key,value>>.");

  // Two-pass plan: construct gather map, and then gather() on structs_column.child(1). Plan A.
  // Can do in one pass perhaps, but that's Plan B.

  auto gather_map              = make_numeric_column(data_type{cudf::type_to_id<size_type>()},
                                        map_column.size(),
                                        mask_state::UNINITIALIZED,
                                        stream,
                                        mr);
  auto gather_map_mutable_view = gather_map->mutable_view();

  auto valid_count = kernel_launcher(map_column, gather_map_mutable_view, lookup_key, stream);

  // Gather map is now available.

  auto values_column    = structs_column.child(1);
  auto table_for_gather = table_view{std::vector<cudf::column_view>{values_column}};

  auto gathered_table = cudf::detail::gather(table_for_gather,
                                             gather_map->view(),
                                             detail::out_of_bounds_policy::IGNORE,
                                             detail::negative_index_policy::NOT_ALLOWED,
                                             mr,
                                             stream);

  return std::make_unique<cudf::column>(std::move(gathered_table->get_column(0)));
}
}  // namespace cudf