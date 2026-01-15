/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

template <typename T>
using hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<T>;
template <typename T>
using hash_value_type = hasher_type<T>::result_type;

template <typename T>
struct set_keys_hasher {
  cudf::column_device_view const d_keys;
  hasher_type<T> hasher{};
  // used by insert
  __device__ hash_value_type<T> operator()(cudf::size_type index) const
  {
    return hasher(d_keys.element<T>(index));
  }
  // used by find
  __device__ hash_value_type<T> operator()(cuda::std::pair<bool, T> const& s) const
  {
    return hasher(s.second);
  }
};

template <typename T>
struct set_keys_equal {
  cudf::column_device_view const d_keys;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const
  {
    if (lhs == rhs) { return true; }
    return d_keys.element<T>(lhs) == d_keys.element<T>(rhs);
  }
  // used by find
  __device__ bool operator()(cuda::std::pair<bool, T> const& lhs, cudf::size_type rhs) const
  {
    return d_keys.element<T>(rhs) == lhs.second;
  }
};

template <typename T, typename SetRef>
CUDF_KERNEL void set_keys_kernel(SetRef set_ref, column_device_view d_input, size_type* d_indices)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx < d_input.size() && d_input.is_valid(idx)) {
    auto const elem = d_input.child(1).element<T>(d_input.element<dictionary32>(idx).value());
    auto const look = cuda::std::pair<bool, T>{true, elem};
    auto const slot = set_ref.find(look);
    d_indices[idx]  = slot != set_ref.end() ? *slot : -1;
  }
}

struct set_keys_dispatch_fn {
  template <typename T>
  std::unique_ptr<column> operator()(dictionary_column_view const& input,
                                     column_view const& new_keys,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_relationally_comparable<T, T>())
  {
    auto d_keys    = column_device_view::create(new_keys, stream);
    auto empty_key = cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
    auto d_equal   = set_keys_equal<T>{*d_keys};
    auto probe     = cuco::linear_probing<1, set_keys_hasher<T>>{set_keys_hasher<T>{*d_keys}};
    auto allocator = rmm::mr::polymorphic_allocator<char>{};
    auto set       = cuco::static_set{
      new_keys.size(), 0.5, empty_key, d_equal, probe, {}, {}, allocator, stream.value()};

    auto const iter = thrust::counting_iterator<cudf::size_type>{0};
    set.insert_async(iter, iter + new_keys.size(), stream.value());

    auto keys_indices = rmm::device_uvector<size_type>(new_keys.size(), stream);
    auto keys_end     = set.retrieve_all(keys_indices.begin(), stream.value());
    keys_indices.resize(cuda::std::distance(keys_indices.begin(), keys_end), stream);

    auto set_ref    = set.ref(cuco::find);
    using set_ref_t = decltype(set_ref);

    // resolve the new indices by using the heterogenous set interface
    auto d_indices = rmm::device_uvector<size_type>(input.size(), stream);
    auto d_input   = column_device_view::create(input.parent(), stream);
    auto grid      = cudf::detail::grid_1d(input.size(), 256);
    set_keys_kernel<T, set_ref_t>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        set_ref, *d_input, d_indices.data());

    // compute the new nulls (existing nulls and any indices < 0)
    auto [null_mask, null_count] = cudf::detail::valid_if(
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(input.size()),
      [d_input = *d_input, d_indices = d_indices.data()] __device__(size_type idx) {
        return (d_input.is_valid(idx) && d_indices[idx] >= 0);
      },
      stream,
      mr);

    // sort the keys_indices so we can use lower-bound on them
    thrust::sort(rmm::exec_policy_nosync(stream), keys_indices.begin(), keys_indices.end());

    // gather the keys using the keys_indices
    auto const oob_policy   = cudf::out_of_bounds_policy::DONT_CHECK;
    auto const index_policy = cudf::detail::negative_index_policy::NOT_ALLOWED;
    auto const keys_tv      = table_view({new_keys});
    auto keys_column =
      std::move(cudf::detail::gather(keys_tv, keys_indices, oob_policy, index_policy, stream, mr)
                  ->release()
                  .front());

    // build the indices column using the same type as the input
    auto indices_column = cudf::make_numeric_column(
      input.indices().type(), input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_result =
      cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());

    // call lower-bound with keys_indices and d_indices to get the new indices values
    auto const begin = keys_indices.begin();
    auto const end   = keys_indices.end();
    thrust::lower_bound(
      rmm::exec_policy_nosync(stream), begin, end, d_indices.begin(), d_indices.end(), d_result);

    // create column with keys_column and indices_column
    return make_dictionary_column(
      std::move(keys_column), std::move(indices_column), std::move(null_mask), null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()(dictionary_column_view const&,
                                     column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref)
    requires(not cudf::is_relationally_comparable<T, T>())
  {
    CUDF_UNREACHABLE("keys are not a valid dictionary key type");
  }
};

}  // namespace

std::unique_ptr<column> set_keys(dictionary_column_view const& input,
                                 column_view const& new_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "keys parameter must not have nulls", std::invalid_argument);
  CUDF_EXPECTS(!new_keys.is_empty(), "keys cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(
    cudf::have_same_types(input.keys(), new_keys), "keys types must match", cudf::data_type_error);

  return type_dispatcher<dispatch_storage_type>(
    new_keys.type(), set_keys_dispatch_fn{}, input, new_keys, stream, mr);
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
