/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

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
  std::unique_ptr<column> operator()(dictionary_column_view const& input,
                                     column_view const& new_keys,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_relationally_comparable<Element, Element>())
  {
    auto dictionary_view = column_device_view::create(input.parent(), stream);
    auto dictionary_itr  = make_dictionary_iterator<Element>(*dictionary_view);
    auto new_keys_view   = column_device_view::create(new_keys, stream);

    auto begin = new_keys_view->begin<Element>();
    auto end   = new_keys_view->end<Element>();

    // create output indices column
    auto result = make_numeric_column(get_indices_type_for_size(new_keys.size()),
                                      input.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr);
    auto result_itr =
      cudf::detail::indexalator_factory::make_output_iterator(result->mutable_view());

#ifdef NDEBUG
    thrust::lower_bound(rmm::exec_policy(stream),
                        begin,
                        end,
                        dictionary_itr,
                        dictionary_itr + input.size(),
                        result_itr,
                        cuda::std::less<Element>());
#else
    // There is a problem with thrust::lower_bound and the output_indexalator
    // https://github.com/NVIDIA/thrust/issues/1452; thrust team created nvbug 3322776
    // This is a workaround.
    thrust::transform(rmm::exec_policy(stream),
                      dictionary_itr,
                      dictionary_itr + input.size(),
                      result_itr,
                      [begin, end] __device__(auto key) {
                        auto itr = thrust::lower_bound(thrust::seq, begin, end, key);
                        return static_cast<size_type>(cuda::std::distance(begin, itr));
                      });
#endif
    result->set_null_count(0);

    return result;
  }

  template <typename Element, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!cudf::is_relationally_comparable<Element, Element>())
  {
    CUDF_FAIL("dictionary set_keys not supported for this column type");
  }
};

}  // namespace

std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& new_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "keys parameter must not have nulls");
  auto keys = dictionary_column.keys();
  CUDF_EXPECTS(
    cudf::have_same_types(keys, new_keys), "keys types must match", cudf::data_type_error);

  // copy the keys -- use cudf::distinct to make sure there are no duplicates,
  // then sort the results.
  auto distinct_keys = cudf::detail::distinct(table_view{{new_keys}},
                                              std::vector<size_type>{0},
                                              duplicate_keep_option::KEEP_ANY,
                                              null_equality::EQUAL,
                                              nan_equality::ALL_EQUAL,
                                              stream,
                                              mr);
  auto sorted_keys   = cudf::detail::sort(distinct_keys->view(),
                                        std::vector<order>{order::ASCENDING},
                                        std::vector<null_order>{null_order::BEFORE},
                                        stream,
                                        mr)
                       ->release();
  std::unique_ptr<column> keys_column(std::move(sorted_keys.front()));

  // compute the new nulls
  auto matches   = cudf::detail::contains(keys_column->view(), keys, stream, mr);
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
                                        stream,
                                        mr);

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(new_nulls.first),
                                new_nulls.second);
}

std::vector<std::unique_ptr<column>> match_dictionaries(
  cudf::host_span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<column_view> keys(input.size());
  std::transform(input.begin(), input.end(), keys.begin(), [](auto& col) { return col.keys(); });
  auto new_keys  = cudf::detail::concatenate(keys, stream, cudf::get_current_device_resource_ref());
  auto keys_view = new_keys->view();
  std::vector<std::unique_ptr<column>> result(input.size());
  std::transform(input.begin(), input.end(), result.begin(), [keys_view, mr, stream](auto& col) {
    return set_keys(col, keys_view, stream, mr);
  });
  return result;
}

std::pair<std::vector<std::unique_ptr<column>>, std::vector<table_view>> match_dictionaries(
  std::vector<table_view> tables, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
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
      auto dict_cols = dictionary::detail::match_dictionaries(dict_views, stream, mr);
      // replace the updated_columns vector entries for the set of columns at col_idx
      auto dict_col_idx = 0;
      for (auto& v : updated_columns)
        v[col_idx] = dict_cols[dict_col_idx++]->view();
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
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_keys(dictionary_column, keys, stream, mr);
}

std::vector<std::unique_ptr<column>> match_dictionaries(
  cudf::host_span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::match_dictionaries(input, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
