/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

namespace cudf::lists {
namespace detail {
namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

// using hash_map      = cuco::static_map<lhs_index_type,
//                                  lhs_index_type,
//                                  cuda::thread_scope_device,
//                                  cudf::detail::hash_table_allocator_type>;
using hash_multimap = cuco::static_multimap<hash_value_type,
                                            lhs_index_type,
                                            cuda::thread_scope_device,
                                            cudf::detail::hash_table_allocator_type>;

using nan_equal_comparator =
  cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;

std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        rmm::cuda_stream_view stream);

std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_rows,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr);
}  // namespace

// This namespace contains code borrow from other WIP PRs, will be removed when they merged.
namespace temporary {

template <typename Comparator>
struct pair_comparator_fn {
  Comparator const d_eqcomp;
  using LHSPair = cuco::pair<hash_value_type, lhs_index_type>;
  using RHSPair = cuco::pair<hash_value_type, rhs_index_type>;

  __device__ bool operator()(LHSPair const& lhs_hash_and_index,
                             RHSPair const& rhs_hash_and_index) const noexcept
  {
    auto const& [lhs_hash, lhs_index] = lhs_hash_and_index;
    auto const& [rhs_hash, rhs_index] = rhs_hash_and_index;
    return lhs_hash == rhs_hash ? d_eqcomp(lhs_index, rhs_index) : false;
  }
};

/**
 * @brief Check the existence of rows in the rhs table in the hash map, which was created by rows
 *        of the lhs table.
 *
 *        Note: This need to be implemented in semi-anti-join
 *        https://github.com/rapidsai/cudf/issues/11037
 */
rmm::device_uvector<bool> check_contains(table_view const& lhs,
                                         table_view const& rhs,
                                         null_equality nulls_equal,
                                         nan_equality nans_equal,
                                         rmm::cuda_stream_view stream)
{
  auto map = std::make_unique<hash_multimap>(
    compute_hash_table_size(lhs.num_rows()),
    cuco::sentinel::empty_key{hash_value_type{cudf::detail::COMPACTION_EMPTY_KEY_SENTINEL}},
    cuco::sentinel::empty_value{lhs_index_type{cudf::detail::COMPACTION_EMPTY_VALUE_SENTINEL}},
    stream.value(),
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream});

  auto const lhs_has_nulls = has_nested_nulls(lhs);
  auto const rhs_has_nulls = has_nested_nulls(rhs);

  // Create a hash map with keys are indices of all elements in the input column.
  // todo: avoid inserting nulls
  {
    auto const hasher   = cudf::experimental::row::hash::row_hasher(lhs, stream);
    auto const d_hasher = cudf::detail::experimental::compaction_hash(
      hasher.device_hasher(nullate::DYNAMIC{lhs_has_nulls}));

    auto const kv_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [d_hasher] __device__(size_type const i) {
        return cuco::make_pair(d_hasher(i), lhs_index_type{i});
      });
    map->insert(kv_it, kv_it + lhs.num_rows(), stream.value());
  }

  auto contained = rmm::device_uvector<bool>(rhs.num_rows(), stream);

  {
    auto const hasher   = cudf::experimental::row::hash::row_hasher(rhs, stream);
    auto const d_hasher = cudf::detail::experimental::compaction_hash(
      hasher.device_hasher(nullate::DYNAMIC{rhs_has_nulls}));

    auto const rhs_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [d_hasher] __device__(size_type const i) {
        return cuco::make_pair(d_hasher(i), rhs_index_type{i});
      });

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(lhs, rhs, stream);

    auto const do_check = [&](auto const& value_comp) {
      auto const d_eqcomp = comparator.equal_to(
        nullate::DYNAMIC{lhs_has_nulls || rhs_has_nulls}, nulls_equal, value_comp);
      map->pair_contains(rhs_it,
                         rhs_it + rhs.num_rows(),
                         contained.begin(),
                         pair_comparator_fn<decltype(d_eqcomp)>{d_eqcomp},
                         stream.value());
    };

    if (nans_equal == nan_equality::ALL_EQUAL) {
      do_check(nan_equal_comparator{});
    } else {
      do_check(nan_unequal_comparator{});
    }
  }

  return contained;
}

/**
 * @brief distinct_map
 *
 * This is the future work: https://github.com/rapidsai/cudf/pull/11052, and
 * https://github.com/rapidsai/cudf/issues/11092
 */
rmm::device_uvector<size_type> distinct_map(
  table_view const& input,
  std::vector<size_type> const& keys,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return rmm::device_uvector<size_type>(0, stream, mr);
  }

  auto const keys_tview = input.select(keys);
  auto const preprocessed_keys =
    cudf::experimental::row::hash::preprocessed_table::create(keys_tview, stream);
  auto const has_null  = nullate::DYNAMIC{cudf::has_nested_nulls(keys_tview)};
  auto const keys_size = keys_tview.num_rows();

  auto key_map = cudf::detail::hash_map_type{
    compute_hash_table_size(keys_size),
    cuco::sentinel::empty_key{cudf::detail::COMPACTION_EMPTY_KEY_SENTINEL},
    cuco::sentinel::empty_value{cudf::detail::COMPACTION_EMPTY_VALUE_SENTINEL},
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value()};

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_keys);
  auto const key_hasher =
    cudf::detail::experimental::compaction_hash(row_hasher.device_hasher(has_null));

  auto const row_comp = cudf::experimental::row::equality::self_comparator(preprocessed_keys);

  auto const kv_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

  auto const do_insert = [&](auto const& value_comp) {
    auto const key_equal = row_comp.equal_to(has_null, nulls_equal, value_comp);
    key_map.insert(kv_iter, kv_iter + input.num_rows(), key_hasher, key_equal, stream.value());
  };

  if (nans_equal == nan_equality::ALL_EQUAL) {
    do_insert(nan_equal_comparator{});
  } else {
    do_insert(nan_unequal_comparator{});
  }

  // The output distinct map.
  auto output_map = rmm::device_uvector<size_type>(key_map.get_size(), stream, mr);
  key_map.retrieve_all(output_map.begin(), thrust::make_discard_iterator(), stream.value());
  return output_map;
}

}  // namespace temporary

namespace {

/**
 * @brief Generate labels for elements in the child column of the input lists column.
 * @param input
 */
std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        rmm::cuda_stream_view stream)
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), cudf::mask_state::UNALLOCATED, stream);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + input.size(), stream);
  return labels;
}

/**
 * @brief Reconstruct an offsets column from the input labels array.
 */
std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_rows,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)

{
  auto out_offsets = make_numeric_column(
    data_type{type_to_id<offset_type>()}, n_rows + 1, mask_state::UNALLOCATED, stream, mr);

  auto const labels_begin  = labels.template begin<size_type>();
  auto const offsets_begin = out_offsets->mutable_view().template begin<size_type>();
  cudf::detail::labels_to_offsets(labels_begin,
                                  labels_begin + labels.size(),
                                  offsets_begin,
                                  offsets_begin + out_offsets->size(),
                                  stream);
  return out_offsets;
}

}  // namespace

std::unique_ptr<column> list_distinct(
  lists_column_view const& input,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const child       = input.get_sliced_child(stream);
  auto const labels      = generate_labels(input, stream);
  auto const input_table = table_view{{labels->view(), child}};

  auto const distinct_indices = temporary::distinct_map(
    table_view{{labels->view(), child}}, {0, 1}, nulls_equal, nans_equal, stream);

  auto index_markers = rmm::device_uvector<bool>(child.size(), stream);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), index_markers.begin(), index_markers.end(), false);
  thrust::scatter(
    rmm::exec_policy(stream),
    thrust::constant_iterator<size_type>(true, 0),
    thrust::constant_iterator<size_type>(true, static_cast<size_type>(distinct_indices.size())),
    distinct_indices.begin(),
    index_markers.begin());

  auto const distinct_table = cudf::detail::copy_if(
    input_table,
    [index_markers = index_markers.begin()] __device__(auto const idx) {
      return index_markers[idx];
    },
    stream,
    mr);

  auto out_offsets =
    reconstruct_offsets(distinct_table->get_column(0).view(), input.size(), stream, mr);

  return make_lists_column(input.size(),
                           std::move(out_offsets),
                           std::move(distinct_table->release().back()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

std::unique_ptr<column> list_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "TBA");

  // - Generate labels for lhs and rhs child elements.
  // - Insert {lhs_labels, lhs_child} table into map.
  // - Check contains for {rhs_labels, rhs_child} table.
  // - `reduce_by_key` with keys are rhs_labels and `logical_or` functor for contains values.

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, stream);
  auto const rhs_labels = generate_labels(rhs, stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  // todo handle nans
  auto const contained =
    temporary::check_contains(lhs_table, rhs_table, nulls_equal, nans_equal, stream);

  // This stores the unique label values, used as scatter map.
  auto list_indices = rmm::device_uvector<size_type>(lhs.size(), stream);

  // Stores the overlap check for non-empty lists.
  auto overlap_result = rmm::device_uvector<bool>(lhs.size(), stream);

  auto const labels_begin           = rhs_labels->view().template begin<size_type>();
  auto const end                    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         labels_begin,  // keys
                                         labels_begin + rhs_labels->size(),  // keys
                                         contained.begin(),  // values to reduce
                                         list_indices.begin(),    // out keys
                                         overlap_result.begin(),  // out values
                                         thrust::equal_to{},  // comp for keys
                                         thrust::logical_or{});  // reduction op for values
  auto const num_non_empty_segments = thrust::distance(overlap_result.begin(), end.second);

  // todo fix null mask null count
  auto result             = make_numeric_column(data_type{type_to_id<bool>()},
                                    lhs.size(),
                                    copy_bitmask(lhs.parent()),  // bitmask and(lhs, rhs)?
                                    lhs.null_count(),
                                    stream,
                                    mr);
  auto const result_begin = result->mutable_view().template begin<bool>();
  thrust::uninitialized_fill(rmm::exec_policy(stream), result_begin, result_begin, false);

  thrust::scatter(rmm::exec_policy(stream),
                  overlap_result.begin(),
                  overlap_result.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  result_begin);

  return result;
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "TBA");

  // - Generate labels for lhs and rhs child elements.
  // - Insert {lhs_labels, lhs_child} table into map.
  // - Check contains for {rhs_labels, rhs_child} table.
  // - copy_if {rhs_indices, rhs_labels} using contains conditions to {gather_map,
  // intersect_labels}.
  // - output_child = pull rhs child elements from gather_map.
  // - output_offsets = reconstruct offsets from intersect_labels.
  // - return lists_column(output_child, output_offsets)

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, stream);
  auto const rhs_labels = generate_labels(rhs, stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  // todo handle nans
  auto const contained =
    temporary::check_contains(lhs_table, rhs_table, nulls_equal, nans_equal, stream);

  auto const intersect_table = cudf::detail::copy_if(
    rhs_table,
    [contained = contained.begin()] __device__(auto const idx) { return contained[idx]; },
    stream);

  // todo: support nans equal
  // todo use detail stream api
  auto const output_table =
    distinct(intersect_table->view(), {0, 1}, nulls_equal, /*nans_equal*/ /*stream*/ mr);

  auto out_offsets =
    reconstruct_offsets(output_table->get_column(0).view(), lhs.size(), stream, mr);

  // todo : fix null
  return make_lists_column(lhs.size(),
                           std::move(out_offsets),
                           std::move(output_table->release().back()),
                           0,
                           {},
                           stream,
                           mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  // - concatenate_row(distinct(lhs), set_except(rhs, lhs))
  // todo: add stream in detail version
  // fix concatenate_rows params.
  auto const lhs_distinct = list_distinct(lhs, nulls_equal, nans_equal, stream);

  // The result table from set_different already contains distinct rows.
  auto const diff = set_difference(rhs, lhs, nulls_equal, nans_equal, mr);

  return lists::concatenate_rows(table_view{{lhs_distinct->view(), diff->view()}},
                                 concatenate_null_policy::IGNORE,
                                 //    stream, //todo: add detail interface
                                 mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "TBA");

  // - Generate labels for lhs and rhs child elements.
  // - Insert {rhs_labels, rhs_child} table into map.
  // - Check contains for {lhs_labels, lhs_child} table.
  // - Invert contains for lhs child element.
  // - copy_if {indices, labels} using the inverted contains conditions to {gather_map,
  //   except_labels} for lhs child elements.
  // - Pull lhs child elements from gather_map.
  // - Reconstruct output offsets from except_labels for lhs.

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, stream);
  auto const rhs_labels = generate_labels(rhs, stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  auto const inv_contained = [&] {
    auto contained =
      temporary::check_contains(rhs_table, lhs_table, nulls_equal, nans_equal, stream);
    thrust::transform(rmm::exec_policy(stream),
                      contained.begin(),
                      contained.end(),
                      contained.begin(),
                      thrust::logical_not{});
    return contained;
  }();

  auto const difference_table = cudf::detail::copy_if(
    lhs_table,
    [inv_contained = inv_contained.begin()] __device__(auto const idx) {
      return inv_contained[idx];
    },
    stream);

  auto const distinct_indices =
    temporary::distinct_map(lhs_table, {0, 1}, nulls_equal, nans_equal, stream);
  auto index_markers = rmm::device_uvector<bool>(lhs_child.size(), stream);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), index_markers.begin(), index_markers.end(), false);
  thrust::scatter(
    rmm::exec_policy(stream),
    thrust::constant_iterator<size_type>(true, 0),
    thrust::constant_iterator<size_type>(true, static_cast<size_type>(distinct_indices.size())),
    distinct_indices.begin(),
    index_markers.begin());

  auto const output_table = cudf::detail::copy_if(
    lhs_table,
    [index_markers = index_markers.begin()] __device__(auto const idx) {
      return index_markers[idx];
    },
    stream,
    mr);

  auto out_offsets =
    reconstruct_offsets(output_table->get_column(0).view(), lhs.size(), stream, mr);

  return make_lists_column(lhs.size(),
                           std::move(out_offsets),
                           std::move(output_table->release().back()),
                           lhs.null_count(),
                           cudf::detail::copy_bitmask(lhs.parent(), stream, mr),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> list_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::list_overlap(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_intersect(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_union(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_difference(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
