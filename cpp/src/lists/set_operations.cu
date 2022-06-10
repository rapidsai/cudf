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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_multimap.cuh>

namespace cudf::lists {
namespace detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;
using hash_map = cuco::static_map<lhs_index_type,
                                  lhs_index_type,
                                  cuda::thread_scope_device,
                                  cudf::detail::hash_table_allocator_type>;

/**
 * @brief Create a hash map with keys are indices of all elements in the input column.
 */
std::unique_ptr<hash_map> create_map(column_view const& input,
                                     null_equality nulls_equal,
                                     rmm::cuda_stream_view stream)
{
  auto map = std::make_unique<hash_map>(
    compute_hash_table_size(input.size()),
    cuco::sentinel::empty_key{cudf::detail::COMPACTION_EMPTY_KEY_SENTINEL},
    cuco::sentinel::empty_value{cudf::detail::COMPACTION_EMPTY_VALUE_SENTINEL},
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value());

  auto const has_nulls = has_nested_nulls(input);
  auto const input_tv  = table_view{{input}};
  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input_tv, stream);

  auto const hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const d_hasher =
    cudf::detail::experimental::compaction_hash(hasher.device_hasher(nullate::DYNAMIC{has_nulls}));

  auto const comparator = cudf::experimental::row::equality::self_comparator(preprocessed_input);
  // todo handle nans
  auto const d_eqcomp = comparator.equal_to(nullate::DYNAMIC{has_nulls}, nulls_equal);

  auto const kv_it =
    cudf::detail::make_counting_transform_iterator(size_type{0}, [] __device__(size_type const i) {
      return cuco::make_pair(lhs_index_type{i}, lhs_index_type{i});
    });
  map->insert(kv_it, kv_it + input.size(), d_hasher, d_eqcomp, stream.value());
  return map;
}

/**
 * @brief Check the existence of rows in the rhs column in the hash map, which was created by rows
 *        of the lhs column.
 */
// todo: keys must be table with the first col is labels
// todo handle nans
auto check_contains(std::unique_ptr<hash_map> const& map,
                    column_view const& lhs,
                    column_view const& rhs,
                    bool const lhs_has_nulls,
                    bool const rhs_has_nulls,
                    null_equality nulls_equal,
                    rmm::cuda_stream_view stream)
{
  auto contained = rmm::device_uvector<bool>(rhs.size(), stream);

  auto const lhs_tv = table_view{{lhs}};
  auto const rhs_tv = table_view{{rhs}};

  auto const hasher   = cudf::experimental::row::hash::row_hasher(rhs_tv, stream);
  auto const d_hasher = cudf::detail::experimental::compaction_hash(
    hasher.device_hasher(nullate::DYNAMIC{rhs_has_nulls}));

  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator(lhs_tv, rhs_tv, stream);
  auto const d_eqcomp =
    comparator.equal_to(nullate::DYNAMIC{lhs_has_nulls || rhs_has_nulls}, nulls_equal);

  auto const rhs_it = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return rhs_index_type{i}; });
  map->contains(rhs_it, rhs_it + rhs.size(), contained.begin(), d_hasher, d_eqcomp, stream.value());
  return contained;
}

/**
 * @brief Generate labels for elements in the child column of the input lists column.
 * @param input
 */
auto generate_labels(lists_column_view const& input, rmm::cuda_stream_view stream)
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
auto reconstruct_offsets(column_view const& labels,
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

std::unique_ptr<column> set_overlap(lists_column_view const& lhs,
                                    lists_column_view const& rhs,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "TBA");

  // - Insert lhs child elements into map.
  // - Check contains for rhs child elements.
  // - Generate labels for rhs child elements.
  // - `reduce_by_key` with `logical_or` functor and keys are labels, values are contains.

  auto const lhs_child           = lhs.get_sliced_child(stream);
  auto const rhs_child           = rhs.get_sliced_child(stream);
  auto const lhs_child_has_nulls = has_nested_nulls(lhs_child);
  auto const rhs_child_has_nulls = has_nested_nulls(rhs_child);

  auto const map = create_map(lhs_child, nulls_equal, stream);
  // todo handle nans
  auto const contained = check_contains(
    map, lhs_child, rhs_child, lhs_child_has_nulls, rhs_child_has_nulls, nulls_equal, stream);
  auto const labels       = generate_labels(rhs_child, stream);
  auto const labels_begin = labels->view().template begin<size_type>();

  // This stores the unique label values, used as scatter map.
  auto list_indices = rmm::device_uvector<size_type>(lhs.size(), stream);

  // Stores the overlap check for non-empty lists.
  auto overlap_result = rmm::device_uvector<bool>(lhs.size(), stream);

  auto const end                    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         labels_begin,                  // keys
                                         labels_begin + labels.size(),  // keys
                                         contained.begin(),  // values to reduce
                                         list_indices.begin(),    // out keys
                                         overlap_result.begin(),  // out values
                                         thrust::equal_to{},  // comp for keys
                                         thrust::logical_or{});  // reduction op for values
  auto const num_non_empty_segments = thrust::distance(overlap_result.begin(), end.second);

  // todo fix null mask null count
  auto result             = make_numeric_column(data_type{type_to_id<bool>()},
                                    lhs.size(),
                                    copy_bitmask(lhs.parent()),
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

  // - Insert lhs child elements into map.
  // - Check contains for rhs child element.
  // - Generate labels for rhs child elements.
  // - copy_if {indices, labels} for rhs child elements using contains conditions to {gather_map,
  //   intersect_labels}.
  // - output_child = pull rhs child elements from gather_map.
  // - output_offsets = reconstruct offsets from intersect_labels.
  // - return lists_column(output_child, output_offsets)

  auto const lhs_child           = lhs.get_sliced_child(stream);
  auto const rhs_child           = rhs.get_sliced_child(stream);
  auto const lhs_child_has_nulls = has_nested_nulls(lhs_child);
  auto const rhs_child_has_nulls = has_nested_nulls(rhs_child);

  auto const map = create_map(lhs_child, nulls_equal, stream);
  // todo handle nans
  auto const contained = check_contains(
    map, lhs_child, rhs_child, lhs_child_has_nulls, rhs_child_has_nulls, nulls_equal, stream);
  auto const labels = generate_labels(rhs_child, stream);

  auto const output_table = cudf::detail::copy_if(
    table_view{{labels->view(), rhs_child}},
    [contained = contained.begin()] __device__(auto const idx) { return contained[idx]; },
    stream,
    mr);

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
  // - concatenate_row(lhs, set_except(rhs, lhs))
  // - Alternative: concatenate_row(lhs, rhs) then `drop_list_duplicates`, however,
  //   `drop_list_duplicates` currently doesn't support nested types.
  // todo: add stream in detail version
  // fix concatenate_rows params.
  auto const diff = set_difference(rhs, lhs, nulls_equal, nans_equal, stream, mr);
  return lists::concatenate_rows(table_view{{lhs.parent(), diff->view()}});
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "TBA");

  // - Insert rhs child elements.
  // - Check contains for lhs child element.
  // - Invert contains for lhs child element.
  // - Generate labels for lhs child elements.
  // - copy_if {indices, labels} using the inverted contains conditions to {gather_map,
  //   except_labels} for lhs child elements.
  // - Pull lhs child elements from gather_map.
  // - Reconstruct output offsets from except_labels for lhs.

  auto const lhs_child           = lhs.get_sliced_child(stream);
  auto const rhs_child           = rhs.get_sliced_child(stream);
  auto const lhs_child_has_nulls = has_nested_nulls(lhs_child);
  auto const rhs_child_has_nulls = has_nested_nulls(rhs_child);

  auto const map           = create_map(rhs_child, nulls_equal, stream);
  auto const inv_contained = [&] {
    auto contained = check_contains(
      map, rhs_child, lhs_child, rhs_child_has_nulls, lhs_child_has_nulls, nulls_equal, stream);
    thrust::transform(rmm::exec_policy(stream),
                      contained.begin(),
                      contained.end(),
                      contained.begin(),
                      thrust::logical_not{});
    return contained;
  }();

  auto const labels = generate_labels(lhs_child, stream);

  auto const output_table = cudf::detail::copy_if(
    table_view{{labels->view(), lhs_child}},
    [inv_contained = inv_contained.begin()] __device__(auto const idx) {
      return inv_contained[idx];
    },
    stream,
    mr);

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

}  // namespace detail

std::unique_ptr<column> set_overlap(lists_column_view const& lhs,
                                    lists_column_view const& rhs,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_overlap(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
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
