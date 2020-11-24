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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/stream_compaction.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/binary_search.h>
#include <algorithm>
#include <iterator>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {
/**
 * @brief Type-dispatch functor for remapping the old indices to new values based on the new
 * key-set.
 *
 * The dispatch is based on the key type.
 * The output column is the new indices column for the new dictionary column.
 */
struct dispatch_compute_indices {
  template <typename Element>
  typename std::enable_if_t<cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(dictionary_column_view const& input,
             column_view const& new_keys,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    auto dictionary_view = column_device_view::create(input.parent(), stream);
    auto d_dictionary    = *dictionary_view;
    auto keys_view       = column_device_view::create(input.keys(), stream);
    auto dictionary_itr  = thrust::make_permutation_iterator(
      keys_view->begin<Element>(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_type>(0), [d_dictionary] __device__(size_type idx) {
          if (d_dictionary.is_null(idx)) return 0;
          return static_cast<size_type>(d_dictionary.element<dictionary32>(idx));
        }));
    auto new_keys_view = column_device_view::create(new_keys, stream);

    // create output indices column
    auto result = make_numeric_column(get_indices_type_for_size(new_keys.size()),
                                      input.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr);
    auto result_itr =
      cudf::detail::indexalator_factory::make_output_iterator(result->mutable_view());
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        new_keys_view->begin<Element>(),
                        new_keys_view->end<Element>(),
                        dictionary_itr,
                        dictionary_itr + input.size(),
                        result_itr,
                        thrust::less<Element>());
    result->set_null_count(0);
    return result;
  }

  template <typename Element>
  typename std::enable_if_t<!cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(dictionary_column_view const& input,
             column_view const& new_keys,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    CUDF_FAIL("list_view dictionary set_keys not supported yet");
  }
};

}  // namespace

//
std::unique_ptr<column> set_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& new_keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "keys parameter must not have nulls");
  auto keys = dictionary_column.keys();
  CUDF_EXPECTS(keys.type() == new_keys.type(), "keys types must match");

  // copy the keys -- use drop_duplicates to make sure they are sorted and unique
  auto table_keys = cudf::detail::drop_duplicates(table_view{{new_keys}},
                                                  std::vector<size_type>{0},
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  stream,
                                                  mr)
                      ->release();
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));

  // compute the new nulls
  auto matches   = cudf::detail::contains(keys, keys_column->view(), stream, mr);
  auto d_matches = matches->view().data<bool>();
  auto indices_itr =
    cudf::detail::indexalator_factory::make_input_iterator(dictionary_column.indices());
  auto d_null_mask = dictionary_column.null_mask();
  auto new_nulls   = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(dictionary_column.offset()),
    thrust::make_counting_iterator<size_type>(dictionary_column.offset() +
                                              dictionary_column.size()),
    [d_null_mask, indices_itr, d_matches] __device__(size_type idx) {
      if (d_null_mask && !bit_is_set(d_null_mask, idx)) return false;
      return d_matches[indices_itr[idx]];
    },
    stream,
    mr);

  // compute the new indices
  auto indices_column = type_dispatcher(keys_column->type(),
                                        dispatch_compute_indices{},
                                        dictionary_column,
                                        keys_column->view(),
                                        mr,
                                        stream);

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(new_nulls.first),
                                new_nulls.second);
}

std::vector<std::unique_ptr<column>> match_dictionaries(std::vector<dictionary_column_view> input,
                                                        rmm::mr::device_memory_resource* mr,
                                                        cudaStream_t stream)
{
  std::vector<column_view> keys(input.size());
  std::transform(input.begin(), input.end(), keys.begin(), [](auto& col) { return col.keys(); });
  auto new_keys  = cudf::detail::concatenate(keys, stream);
  auto keys_view = new_keys->view();
  std::vector<std::unique_ptr<column>> result(input.size());
  std::transform(input.begin(), input.end(), result.begin(), [keys_view, mr, stream](auto& col) {
    return set_keys(col, keys_view, mr, stream);
  });
  return result;
}

std::pair<std::vector<std::unique_ptr<column>>, std::vector<table_view>> match_dictionaries(
  std::vector<table_view> tables, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // Make a copy of all the column views from each table_view
  std::vector<std::vector<column_view>> updated_columns;
  std::transform(tables.begin(), tables.end(), std::back_inserter(updated_columns), [](auto& t) {
    return std::vector<column_view>(t.begin(), t.end());
  });

  // Each column in a table must match in type.
  // Once a dictionary column is found, all the corresponding column_views in the
  // other table_views are matched. The matched column_views then replace the originals.
  std::vector<std::unique_ptr<column>> dictionary_columns;
  auto first_table = tables.front();
  for (size_type col_idx = 0; col_idx < first_table.num_columns(); ++col_idx) {
    auto col = first_table.column(col_idx);
    if (col.type().id() == type_id::DICTIONARY32) {
      std::vector<dictionary_column_view> dict_views;  // hold all column_views at col_idx
      std::transform(
        tables.begin(), tables.end(), std::back_inserter(dict_views), [col_idx](auto& t) {
          return dictionary_column_view(t.column(col_idx));
        });
      // now match the keys in these dictionary columns
      auto dict_cols = dictionary::detail::match_dictionaries(dict_views, mr, stream);
      // replace the updated_columns vector entries for the set of columns at col_idx
      auto dict_col_idx = 0;
      for (auto& v : updated_columns) v[col_idx] = dict_cols[dict_col_idx++]->view();
      // move the updated dictionary columns into the main output vector
      std::move(dict_cols.begin(), dict_cols.end(), std::back_inserter(dictionary_columns));
    }
  }
  // All the new column_views are in now updated_columns.

  // Rebuild the table_views from the column_views.
  std::vector<table_view> updated_tables;
  std::transform(updated_columns.begin(),
                 updated_columns.end(),
                 std::back_inserter(updated_tables),
                 [](auto& v) { return table_view{v}; });

  // Return the new dictionary columns and table_views
  return {std::move(dictionary_columns), std::move(updated_tables)};
}

}  // namespace detail

// external API

std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_keys(dictionary_column, keys, mr);
}

}  // namespace dictionary
}  // namespace cudf
