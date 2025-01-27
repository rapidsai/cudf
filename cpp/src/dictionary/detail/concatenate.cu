/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <vector>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief Keys and indices offsets values.
 *
 * The first value is the keys offsets and the second values is the indices offsets.
 * These are offsets to the beginning of each input column after concatenating.
 */
using offsets_pair = thrust::pair<size_type, size_type>;

/**
 * @brief Utility for calculating the offsets for the concatenated child columns
 *        of the output dictionary column.
 */
struct compute_children_offsets_fn {
  /**
   * @brief Create the utility functor.
   *
   * The columns vector is converted into vector of column_view pointers so they
   * can be used in thrust::transform_exclusive_scan without causing the
   * compiler warning/error: "host/device function calling host function".
   *
   * @param columns The input dictionary columns.
   */
  compute_children_offsets_fn(host_span<column_view const> columns) : columns_ptrs{columns.size()}
  {
    std::transform(
      columns.begin(), columns.end(), columns_ptrs.begin(), [](auto& cv) { return &cv; });
  }

  /**
   * @brief Return the first keys() of the dictionary columns.
   */
  column_view get_keys()
  {
    auto const view(*std::find_if(
      columns_ptrs.begin(), columns_ptrs.end(), [](auto pcv) { return pcv->size() > 0; }));
    return dictionary_column_view(*view).keys();
  }

  /**
   * @brief Create the offsets pair for the concatenated columns.
   *
   * Both vectors have the length of the number of input columns.
   * The sizes of each child (keys and indices) of the individual columns
   * are used to create the offsets.
   *
   * @param stream Stream used for allocating the output rmm::device_uvector.
   * @return Vector of offsets_pair objects for keys and indices.
   */
  rmm::device_uvector<offsets_pair> create_children_offsets(rmm::cuda_stream_view stream)
  {
    auto offsets = cudf::detail::make_host_vector<offsets_pair>(columns_ptrs.size(), stream);
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
    return cudf::detail::make_device_uvector_sync(
      offsets, stream, cudf::get_current_device_resource_ref());
  }

 private:
  std::vector<column_view const*> columns_ptrs;  ///< pointer version of input column_view vector
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
  std::enable_if_t<cudf::is_relationally_comparable<Element, Element>(), std::unique_ptr<column>>
  operator()(column_view const& all_keys,
             column_view const& all_indices,
             column_view const& new_keys,
             offsets_pair const* d_offsets,
             size_type const* d_map_to_keys,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
  {
    auto keys_view     = column_device_view::create(all_keys, stream);
    auto indices_view  = column_device_view::create(all_indices, stream);
    auto d_all_indices = *indices_view;

    auto indices_itr = cudf::detail::indexalator_factory::make_input_iterator(all_indices);
    // map the concatenated indices to the concatenated keys
    auto all_itr = thrust::make_permutation_iterator(
      keys_view->begin<Element>(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0),
        cuda::proclaim_return_type<size_type>(
          [d_offsets, d_map_to_keys, d_all_indices, indices_itr] __device__(size_type idx) {
            if (d_all_indices.is_null(idx)) return 0;
            return indices_itr[idx] + d_offsets[d_map_to_keys[idx]].first;
          })));

    auto new_keys_view = column_device_view::create(new_keys, stream);

    auto begin = new_keys_view->begin<Element>();
    auto end   = new_keys_view->end<Element>();

    // create the indices output column
    auto result = make_numeric_column(
      all_indices.type(), all_indices.size(), mask_state::UNALLOCATED, stream, mr);
    auto result_itr =
      cudf::detail::indexalator_factory::make_output_iterator(result->mutable_view());
    // new indices values are computed by matching the concatenated keys to the new key set

#ifdef NDEBUG
    thrust::lower_bound(rmm::exec_policy(stream),
                        begin,
                        end,
                        all_itr,
                        all_itr + all_indices.size(),
                        result_itr,
                        thrust::less<Element>());
#else
    // There is a problem with thrust::lower_bound and the output_indexalator.
    // https://github.com/NVIDIA/thrust/issues/1452; thrust team created nvbug 3322776
    // This is a workaround.
    thrust::transform(rmm::exec_policy(stream),
                      all_itr,
                      all_itr + all_indices.size(),
                      result_itr,
                      [begin, end] __device__(auto key) {
                        auto itr = thrust::lower_bound(thrust::seq, begin, end, key);
                        return static_cast<size_type>(thrust::distance(begin, itr));
                      });
#endif
    return result;
  }

  template <typename Element, typename... Args>
  std::enable_if_t<!cudf::is_relationally_comparable<Element, Element>(), std::unique_ptr<column>>
  operator()(Args&&...)
  {
    CUDF_FAIL("dictionary concatenate not supported for this column type");
  }
};

}  // namespace

std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  // exception here is the same behavior as in cudf::concatenate
  CUDF_EXPECTS(not columns.empty(), "Unexpected empty list of columns to concatenate.");

  // concatenate the keys (and check the keys match)
  compute_children_offsets_fn child_offsets_fn{columns};
  auto expected_keys = child_offsets_fn.get_keys();
  std::vector<column_view> keys_views(columns.size());
  std::transform(columns.begin(), columns.end(), keys_views.begin(), [expected_keys](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    // empty column may not have keys so we create an empty column_view place-holder
    if (dict_view.is_empty()) return column_view{expected_keys.type(), 0, nullptr, nullptr, 0};
    auto keys = dict_view.keys();
    CUDF_EXPECTS(cudf::have_same_types(keys, expected_keys),
                 "key types of all dictionary columns must match",
                 cudf::data_type_error);
    return keys;
  });
  auto all_keys =
    cudf::detail::concatenate(keys_views, stream, cudf::get_current_device_resource_ref());

  // sort keys and remove duplicates;
  // this becomes the keys child for the output dictionary column
  auto table_keys  = cudf::detail::distinct(table_view{{all_keys->view()}},
                                           std::vector<size_type>{0},
                                           duplicate_keep_option::KEEP_ANY,
                                           null_equality::EQUAL,
                                           nan_equality::ALL_EQUAL,
                                           stream,
                                           mr);
  auto sorted_keys = cudf::detail::sort(table_keys->view(),
                                        std::vector<order>{order::ASCENDING},
                                        std::vector<null_order>{null_order::BEFORE},
                                        stream,
                                        mr)
                       ->release();
  std::unique_ptr<column> keys_column(std::move(sorted_keys.front()));

  // next, concatenate the indices
  std::vector<column_view> indices_views(columns.size());
  std::transform(columns.begin(), columns.end(), indices_views.begin(), [](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    if (dict_view.is_empty()) {
      return column_view{data_type{type_id::INT32}, 0, nullptr, nullptr, 0};
    }
    return dict_view.get_indices_annotated();  // nicely includes validity mask and view offset
  });
  auto all_indices        = cudf::detail::concatenate(indices_views, stream, mr);
  auto const indices_size = all_indices->size();

  // build a vector of values to map the old indices to the concatenated keys
  auto children_offsets = child_offsets_fn.create_children_offsets(stream);
  rmm::device_uvector<size_type> map_to_keys(indices_size, stream);
  auto indices_itr = cudf::detail::make_counting_transform_iterator(
    1, cuda::proclaim_return_type<offsets_pair>([] __device__(size_type idx) {
      return offsets_pair{0, idx};
    }));
  // the indices offsets (pair.second) are for building the map
  thrust::lower_bound(
    rmm::exec_policy(stream),
    children_offsets.begin() + 1,
    children_offsets.end(),
    indices_itr,
    indices_itr + indices_size,
    map_to_keys.begin(),
    [] __device__(auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; });

  // now recompute the indices values for the new keys_column;
  // the keys offsets (pair.first) are for mapping to the input keys
  auto indices_column = type_dispatcher(expected_keys.type(),
                                        dispatch_compute_indices{},
                                        all_keys->view(),     // old keys
                                        all_indices->view(),  // old indices
                                        keys_column->view(),  // new keys
                                        children_offsets.data(),
                                        map_to_keys.data(),
                                        stream,
                                        mr);

  // remove the bitmask from the all_indices
  auto null_count = all_indices->null_count();  // get before release()
  auto contents   = all_indices->release();     // all_indices will now be empty

  // finally, frankenstein that dictionary column together
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(contents.null_mask.release())),
                                null_count);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
