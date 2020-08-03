/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <algorithm>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

// keys and indices offsets for concatenated child columns
using offsets_pair = thrust::pair<size_type, size_type>;

struct compute_child_offsets {
  compute_child_offsets(std::vector<column_view> const& columns) : columns_ptrs{columns.size()}
  {
    std::transform(
      columns.begin(), columns.end(), columns_ptrs.begin(), [](auto& cv) { return &cv; });
  }

  data_type get_keys_type()
  {
    dictionary_column_view dict_view(**std::find_if(
      columns_ptrs.begin(), columns_ptrs.end(), [](auto pcv) { return pcv->size() > 0; }));
    return dict_view.keys().type();
  }

  rmm::device_uvector<offsets_pair> create_children_offsets(cudaStream_t stream)
  {
    std::vector<offsets_pair> offsets(columns_ptrs.size());
    thrust::transform_exclusive_scan(
      thrust::host,
      columns_ptrs.begin(),
      columns_ptrs.end(),
      offsets.begin(),
      [](auto pcv) {
        dictionary_column_view view(*pcv);
        return offsets_pair{view.keys_size(), view.size()};
      },
      offsets_pair{0, 0},
      [](auto lhs, auto rhs) {
        return offsets_pair{lhs.first + rhs.first, lhs.second + rhs.second};
      });
    auto d_offsets = rmm::device_uvector<offsets_pair>(offsets.size(), stream);
    CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                             offsets.data(),
                             offsets.size() * sizeof(offsets_pair),
                             cudaMemcpyHostToDevice,
                             stream));
    return d_offsets;
  }

 private:
  // thrust::transform_exclusive_scan does not compile with column_view
  std::vector<column_view const*> columns_ptrs;
};

/**
 * @brief Type-dispatch functor for remapping the old indices to new values based
 * on the new key-set.
 *
 * The dispatch is based on the key type.
 * The output column is the updated indices child for the new dictionary column.
 */
struct dispatch_compute_indices {
  template <typename Element>
  typename std::enable_if_t<cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(column_view const& all_keys,
             column_view const& all_indices,
             column_view const& new_keys,
             offsets_pair const* d_offsets,
             size_type const* d_map_to_keys,
             cudaStream_t stream,
             rmm::mr::device_memory_resource* mr)
  {
    auto keys_view     = column_device_view::create(all_keys, stream);
    auto d_all_keys    = *keys_view;
    auto indices_view  = column_device_view::create(all_indices, stream);
    auto d_all_indices = *indices_view;
    auto all_itr       = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      [d_all_keys, d_offsets, d_map_to_keys, d_all_indices] __device__(size_type idx) {
        if (d_all_indices.is_null(idx)) return Element{};
        size_type index =
          d_all_indices.template element<int32_t>(idx) + d_offsets[d_map_to_keys[idx]].first;
        return d_all_keys.template element<Element>(index);
      });
    auto new_keys_view = column_device_view::create(new_keys, stream);
    auto d_new_keys    = *new_keys_view;
    auto keys_itr      = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      [d_new_keys] __device__(size_type idx) { return d_new_keys.template element<Element>(idx); });

    auto result = make_numeric_column(
      data_type{type_id::INT32}, all_indices.size(), mask_state::UNALLOCATED, stream, mr);
    auto d_result = result->mutable_view().data<int32_t>();
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        keys_itr,
                        keys_itr + new_keys.size(),
                        all_itr,
                        all_itr + all_indices.size(),
                        d_result,
                        thrust::less<Element>());
    result->set_null_count(0);
    return result;
  }

  template <typename Element>
  typename std::enable_if_t<!cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(column_view const&,
             column_view const&,
             column_view const&,
             offsets_pair const*,
             size_type const*,
             cudaStream_t stream,
             rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("list_view as keys for dictionary not supported");
  }
};

}  // namespace

std::unique_ptr<column> concatenate(std::vector<column_view> const& columns,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  if (columns.size() == 0) return make_empty_column(data_type{type_id::DICTIONARY32});

  // concatenate the keys (and check the keys match)
  compute_child_offsets child_offsets_fn{columns};
  auto keys_type = child_offsets_fn.get_keys_type();
  std::vector<column_view> keys_views(columns.size());
  std::transform(columns.begin(), columns.end(), keys_views.begin(), [keys_type](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    if (dict_view.size() == 0) return column_view{keys_type, 0, nullptr};
    auto keys = dict_view.keys();
    CUDF_EXPECTS(keys.type() == keys_type, "key types of each dictionary column must match");
    return keys;
  });
  auto all_keys = cudf::detail::concatenate(keys_views, rmm::mr::get_default_resource(), stream);

  // sort keys and remove duplicates
  auto table_keys = cudf::detail::drop_duplicates(table_view{{all_keys->view()}},
                                                  std::vector<size_type>{0},
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  mr,
                                                  stream)
                      ->release();
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));

  // concatenate the indices
  std::vector<column_view> indices_views(columns.size());
  std::transform(columns.begin(), columns.end(), indices_views.begin(), [](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    if (dict_view.size() == 0) return column_view{data_type{type_id::INT32}, 0, nullptr};
    return dict_view.get_indices_annotated();  // includes validity mask and offset
  });
  auto all_indices       = cudf::detail::concatenate(indices_views, mr, stream);
  size_type indices_size = all_indices->size();

  // create vector of values to match old indices to the concatenated keys
  auto children_offsets = child_offsets_fn.create_children_offsets(stream);

  rmm::device_uvector<size_type> map_to_keys(indices_size, stream);
  auto pair_itr = thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(1),
                                                  [] __device__(size_type idx) {
                                                    return offsets_pair{0, idx};
                                                  });
  thrust::lower_bound(
    rmm::exec_policy(stream)->on(stream),
    children_offsets.begin() + 1,
    children_offsets.end(),
    pair_itr,
    pair_itr + indices_size + 1,
    map_to_keys.begin(),
    [] __device__(auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; });

  // now recompute the indices values for the new keys_column
  auto indices_column = type_dispatcher(keys_type,
                                        dispatch_compute_indices{},
                                        all_keys->view(),     // old keys
                                        all_indices->view(),  // old indices
                                        keys_column->view(),  // new keys
                                        children_offsets.data(),
                                        map_to_keys.data(),
                                        stream,
                                        mr);

  // remove the bitmask from the indices_column
  auto null_count = all_indices->null_count();  // get before release()
  auto contents   = all_indices->release();     // all_indices will now be empty

  // finally, frankenstein the dictionary column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(contents.null_mask.release())),
                                null_count);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
