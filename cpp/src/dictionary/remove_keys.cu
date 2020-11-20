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

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief Return a new dictionary by removing identified keys from the provided dictionary.
 *
 * This is a common utility for `remove_keys` and `remove_unused_keys` detail functions.
 * It will create a new dictionary with the remaining keys and create new indices values
 * to go with these new keys.
 *
 * @tparam KeysKeeper Function bool(size_type) that takes keys position index
 *                    and returns true if that key is to be used in the output dictionary.
 * @param dictionary_column The column to use for creating the new dictionary.
 * @param keys_to_keep_fn Called to determine which keys in `dictionary_column` to keep.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename KeysKeeper>
std::unique_ptr<column> remove_keys_fn(
  dictionary_column_view const& dictionary_column,
  KeysKeeper keys_to_keep_fn,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  auto const keys_view    = dictionary_column.keys();
  auto execpol            = rmm::exec_policy(stream);
  auto const indices_type = dictionary_column.indices().type();
  auto const max_size     = dictionary_column.size();

  // create/init indices map array
  auto map_indices =
    make_fixed_width_column(indices_type, keys_view.size(), mask_state::UNALLOCATED, stream);
  auto map_itr =
    cudf::detail::indexalator_factory::make_output_iterator(map_indices->mutable_view());
  // init to max to identify new nulls
  thrust::fill(execpol->on(stream),
               map_itr,
               map_itr + keys_view.size(),
               max_size);  // all valid indices are less than this value

  // build keys column and indices map
  std::unique_ptr<column> keys_column = [&] {
    // create keys positions column to identify original key positions after removing they keys
    auto keys_positions = [&] {
      auto positions = make_fixed_width_column(
        indices_type, keys_view.size(), cudf::mask_state::UNALLOCATED, stream);
      auto itr = cudf::detail::indexalator_factory::make_output_iterator(positions->mutable_view());
      thrust::sequence(execpol->on(stream), itr, itr + keys_view.size());
      return positions;
    }();
    // copy the non-removed keys ( keys_to_keep_fn(idx)==true )
    auto table_keys =
      cudf::detail::copy_if(
        table_view{{keys_view, keys_positions->view()}}, keys_to_keep_fn, stream, mr)
        ->release();
    auto const filtered_view = table_keys[1]->view();
    auto filtered_itr = cudf::detail::indexalator_factory::make_input_iterator(filtered_view);
    auto positions_itr =
      cudf::detail::indexalator_factory::make_input_iterator(keys_positions->view());
    // build indices mapper
    // Example scatter([0,1,2][0,2,4][max,max,max,max,max]) => [0,max,1,max,2]
    thrust::scatter(execpol->on(stream),
                    positions_itr,
                    positions_itr + filtered_view.size(),
                    filtered_itr,
                    map_itr);
    return std::move(table_keys.front());
  }();

  // create non-nullable indices view with offset applied -- this is used as a gather map
  column_view indices_view(dictionary_column.indices().type(),
                           dictionary_column.size(),
                           dictionary_column.indices().head(),
                           nullptr,
                           0,
                           dictionary_column.offset());
  // create new indices column
  // Example: gather([0,max,1,max,2],[4,0,3,1,2,2,2,4,0]) => [2,0,max,max,1,1,1,2,0]
  auto table_indices = cudf::detail::gather(table_view{{map_indices->view()}},
                                            indices_view,
                                            cudf::detail::out_of_bounds_policy::NULLIFY,
                                            cudf::detail::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr)
                         ->release();
  std::unique_ptr<column> indices_column(std::move(table_indices.front()));

  // compute new nulls -- merge the existing nulls with the newly created ones (value<0)
  auto const offset = dictionary_column.offset();
  auto d_null_mask  = dictionary_column.null_mask();
  auto indices_itr = cudf::detail::indexalator_factory::make_input_iterator(indices_column->view());
  auto new_nulls   = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(dictionary_column.size()),
    [offset, d_null_mask, indices_itr, max_size] __device__(size_type idx) {
      if (d_null_mask && !bit_is_set(d_null_mask, idx + offset)) return false;
      return (indices_itr[idx] < max_size);  // new nulls have max values
    },
    stream,
    mr);
  rmm::device_buffer new_null_mask =
    (new_nulls.second > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};

  // create column with keys_column and indices_column
  return make_dictionary_column(
    std::move(keys_column), std::move(indices_column), std::move(new_null_mask), new_nulls.second);
}

}  // namespace

std::unique_ptr<column> remove_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& keys_to_remove,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(!keys_to_remove.has_nulls(), "keys_to_remove must not have nulls");
  auto const keys_view = dictionary_column.keys();
  CUDF_EXPECTS(keys_view.type() == keys_to_remove.type(), "keys types must match");

  // locate keys to remove by searching the keys column
  auto const matches = cudf::detail::contains(keys_view, keys_to_remove, stream, mr);
  auto d_matches     = matches->view().data<bool>();
  // call common utility method to keep the keys not matched to keys_to_remove
  auto key_matcher = [d_matches] __device__(size_type idx) { return !d_matches[idx]; };
  return remove_keys_fn(dictionary_column, key_matcher, mr, stream);
}

std::unique_ptr<column> remove_unused_keys(
  dictionary_column_view const& dictionary_column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  // locate the keys to remove
  auto const keys_size     = dictionary_column.keys_size();
  column_view indices_view = dictionary_column.get_indices_annotated();

  // search the indices values with key indices to look for any holes
  auto const matches = [&] {
    // build keys index to verify against indices values
    rmm::device_uvector<uint32_t> keys_positions(keys_size, stream);
    thrust::sequence(
      rmm::exec_policy(stream)->on(stream), keys_positions.begin(), keys_positions.end());
    // wrap the indices for comparison in contains()
    column_view keys_positions_view(data_type{type_id::UINT32}, keys_size, keys_positions.data());
    return cudf::detail::contains(keys_positions_view, indices_view, stream, mr);
  }();
  auto d_matches = matches->view().data<bool>();

  // call common utility method to keep the keys that match
  auto key_matcher = [d_matches] __device__(size_type idx) { return d_matches[idx]; };
  return remove_keys_fn(dictionary_column, key_matcher, mr, stream);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> remove_keys(dictionary_column_view const& dictionary_column,
                                    column_view const& keys_to_remove,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::remove_keys(dictionary_column, keys_to_remove, mr);
}

std::unique_ptr<column> remove_unused_keys(dictionary_column_view const& dictionary_column,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::remove_unused_keys(dictionary_column, mr);
}

}  // namespace dictionary
}  // namespace cudf
