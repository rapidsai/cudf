/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <stream_compaction/drop_duplicates.cuh>
#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <vector>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Create a column_view of index values which represent the row values
 * without duplicates as per @p `keep`
 *
 * Given a `keys` table_view, each row index is copied to output `unique_indices`, if the
 * corresponding row of `keys` table_view is unique, where the definition of unique depends on the
 * value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: only unique rows are kept
 *
 * @param[in] keys            table_view to identify duplicate rows
 * @param[out] unique_indices Column to store the index with unique rows
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_equal     flag to denote nulls are equal if null_equality::EQUAL,
 * @param[in] null_precedence flag to denote nulls should appear before or after non-null items,
 *                            nulls are not equal if null_equality::UNEQUAL
 * @param[in] stream          CUDA stream used for device memory operations and kernel launches.
 *
 * @return column_view column_view of unique row index as per specified `keep`, this is actually
 * slice of `unique_indices`.
 */
column_view get_unique_ordered_indices(cudf::table_view const& keys,
                                       cudf::mutable_column_view& unique_indices,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       null_order null_precedence,
                                       rmm::cuda_stream_view stream)
{
  // Sort only the indices.
  // Note that stable sort must be used to maintain the order of duplicate elements.
  auto sorted_indices = stable_sorted_order(
    keys,
    std::vector<order>{},
    std::vector<null_order>{static_cast<uint64_t>(keys.num_columns()), null_precedence},
    stream,
    rmm::mr::get_current_device_resource());

  // extract unique indices
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  auto comp = row_equality_comparator(
    nullate::DYNAMIC{cudf::has_nulls(keys)}, *device_input_table, *device_input_table, nulls_equal);
  auto result_end = unique_copy(sorted_indices->view().begin<cudf::size_type>(),
                                sorted_indices->view().end<cudf::size_type>(),
                                unique_indices.begin<cudf::size_type>(),
                                comp,
                                keep,
                                stream);

  return cudf::detail::slice(column_view(unique_indices),
                             0,
                             thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
}
}  // namespace

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0 or input.num_columns() == 0 or keys.empty()) { return empty_like(input); }

  auto unique_indices =
    make_numeric_column(data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_view     = mutable_column_device_view::create(*unique_indices, stream);
  auto keys_view        = input.select(keys);
  auto keys_device_view = cudf::table_device_view::create(keys_view, stream);
  auto row_equal        = row_equality_comparator(nullate::DYNAMIC{cudf::has_nulls(keys_view)},
                                           *keys_device_view,
                                           *keys_device_view,
                                           nulls_equal);

  // get indices of unique rows
  auto result_end = unique_copy(thrust::counting_iterator<size_type>(0),
                                thrust::counting_iterator<size_type>(num_rows),
                                mutable_view->begin<size_type>(),
                                row_equal,
                                keep,
                                stream);
  auto indices_view =
    cudf::detail::slice(column_view(*unique_indices),
                        0,
                        thrust::distance(mutable_view->begin<size_type>(), result_end));

  // gather unique rows and return
  return detail::gather(input,
                        indices_view,
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

std::unique_ptr<table> sort_and_drop_duplicates(table_view const& input,
                                                std::vector<size_type> const& keys,
                                                duplicate_keep_option keep,
                                                null_equality nulls_equal,
                                                null_order null_precedence,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);

  // The values will be filled into this column
  auto unique_indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, keys_view.num_rows(), mask_state::UNALLOCATED, stream);
  auto mutable_unique_indices_view = unique_indices->mutable_view();
  // This is just slice of `unique_indices` but with different size as per the
  // keys_view has been processed in `get_unique_ordered_indices`
  auto unique_indices_view = detail::get_unique_ordered_indices(
    keys_view, mutable_unique_indices_view, keep, nulls_equal, null_precedence, stream);

  // run gather operation to establish new order
  return detail::gather(input,
                        unique_indices_view,
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

std::unique_ptr<table> unordered_drop_duplicates(table_view const& input,
                                                 std::vector<size_type> const& keys,
                                                 null_equality nulls_equal,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  if (0 == input.num_rows() || 0 == input.num_columns() || 0 == keys.size()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);
  auto table_ptr = cudf::table_device_view::create(keys_view, stream);
  auto has_null  = nullate::DYNAMIC{cudf::has_nulls(keys_view)};
  auto const num_rows{table_ptr->num_rows()};

  hash_map_type key_map{compute_hash_table_size(num_rows),
                        COMPACTION_EMPTY_KEY_SENTINEL,
                        COMPACTION_EMPTY_VALUE_SENTINEL,
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};

  compaction_hash hash_key{has_null, *table_ptr};
  row_equality_comparator row_equal(has_null, *table_ptr, *table_ptr, nulls_equal);

  auto iter = cudf::detail::make_counting_transform_iterator(0, [] __device__(size_type i) {
    // TODO: cuco::make_pair currently requires rvalue references. We
    // create a copy to avoid double-move invoking undefined behavior.
    auto ii = i;
    return cuco::make_pair(std::move(i), std::move(ii));
  });
  // insert unique indices into the map.
  key_map.insert(iter, iter + num_rows, hash_key, row_equal, stream.value());

  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  rmm::device_uvector<bool> index_exists_in_map(num_rows, stream, mr);
  // enumerate all indices to check if they are present in the map.
  key_map.contains(counting_iter, counting_iter + num_rows, index_exists_in_map.begin(), hash_key);

  auto const output_size{key_map.get_size()};

  // write unique indices to a numeric column
  auto unique_indices = make_numeric_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr);
  auto mutable_view = mutable_column_device_view::create(*unique_indices, stream);
  thrust::copy_if(rmm::exec_policy(stream),
                  counting_iter,
                  counting_iter + num_rows,
                  index_exists_in_map.begin(),
                  mutable_view->begin<size_type>(),
                  thrust::identity<bool>{});

  // run gather operation to establish new order
  return detail::gather(input,
                        unique_indices->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option const keep,
                                       null_equality nulls_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_duplicates(input, keys, keep, nulls_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> sort_and_drop_duplicates(table_view const& input,
                                                std::vector<size_type> const& keys,
                                                duplicate_keep_option const keep,
                                                null_equality nulls_equal,
                                                null_order null_precedence,
                                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_and_drop_duplicates(
    input, keys, keep, nulls_equal, null_precedence, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> unordered_drop_duplicates(table_view const& input,
                                                 std::vector<size_type> const& keys,
                                                 null_equality nulls_equal,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::unordered_drop_duplicates(input, keys, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
