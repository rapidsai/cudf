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

/**
 * @brief Device function that searches for the specified lookup_key
 * in the list at index `row_index`, and writes out the index of the 
 * first match to the output.
 * 
 * This function is called once per row of the `input` column
 * If the lookup_key is not found, (-1) is returned for that list row.
 */
template <bool has_nulls>
void __device__ search_each_list(size_type row_index,
                                 column_device_view input,
                                 mutable_column_device_view output,
                                 string_scalar_device_view lookup_key)
{
  if (has_nulls && input.is_null(row_index)) {  // List row is null.
    output.element<size_type>(row_index) = -1;  // Not found.
    return;
  }

  auto offsets{input.child(0)};
  auto start_index{offsets.element<size_type>(row_index)};
  auto end_index{offsets.element<size_type>(row_index + 1)};

  auto key_column{input.child(1).child(0)};

  for (size_type list_element_index{start_index}; list_element_index < end_index;
       ++list_element_index) {
    if (has_nulls && key_column.is_null(list_element_index)) {
      continue; // Skip the list-element with null-key.
    }

    // List element's key is not null. Check if it matches the lookup_key.
    if (key_column.element<string_view>(list_element_index) == lookup_key.value()) {
      output.element<size_type>(row_index) = list_element_index;
      return;
    }
  }

  output.element<size_type>(row_index) = -1;  // Not found.
}

/**
 * @brief The map-lookup CUDA kernel, which searches for the specified `lookup_key`
 * string in each list<string> row of the `input` column.
 *
 * The kernel writes the index (into the `input` list-column's child) where the `lookup_key`
 * is found, to the `output` column. If the `lookup_key` is not found, (-1) is written instead. 
 *
 * The produces one output row per input, with no nulls. The output may then be used
 * with `cudf::gather()`, to find the values corresponding to the `lookup_key`.
 */
template <int block_size, bool has_nulls>
__launch_bounds__(block_size) __global__ void gpu_find_first(column_device_view input,
                                                             mutable_column_device_view output,
                                                             string_scalar_device_view lookup_key)
{
  size_type tid      = blockIdx.x * block_size + threadIdx.x;
  size_type stride = block_size * gridDim.x;

  // Each CUDA thread processes one row of `input`. Each row is a list<string>.
  // So each thread searches for `lookup_key` in one row of the input column,
  // and writes its index out to output.
  while (tid < input.size()) {
    search_each_list<has_nulls>(tid, input, output, lookup_key);
    tid += stride;
  }
}

/**
 * @brief Function to generate a gather-map, based on the location of the `lookup_key`
 * string in each row of the input.
 *
 * The gather map may then be used to gather the values corresponding to the `lookup_key`
 * for each row.
 */
template <bool has_nulls>
std::unique_ptr<column> get_gather_map_for_map_values(column_view const& input,
                                                      string_scalar& lookup_key,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{
  constexpr size_type block_size{256};
  cudf::detail::grid_1d grid{input.size(), block_size};

  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto lookup_key_device_view{get_scalar_device_view(lookup_key)};
  auto gather_map = make_numeric_column(
    data_type{cudf::type_to_id<size_type>()}, input.size(), mask_state::ALL_VALID, stream, mr);
  auto output_view = mutable_column_device_view::create(gather_map->mutable_view(), stream);

  gpu_find_first<block_size, has_nulls><<<grid.num_blocks, block_size, 0, stream>>>(
    *input_device_view, *output_view, lookup_key_device_view);

  CHECK_CUDA(stream);

  return gather_map;
}

}  // namespace

namespace jni {
std::unique_ptr<column> map_lookup(column_view const& map_column,
                                   string_scalar lookup_key,
                                   bool has_nulls,
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
  // (Can do in one pass perhaps, but that's Plan B.)

  auto gather_map = has_nulls? 
     get_gather_map_for_map_values<true>(map_column, lookup_key, mr, stream)
   : get_gather_map_for_map_values<false>(map_column, lookup_key, mr, stream);

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
} // namespace jni;
} // namespace cudf;