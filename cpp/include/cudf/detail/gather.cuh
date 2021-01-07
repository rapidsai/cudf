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
#pragma once

#include <cudf/detail/copy.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

namespace cudf {
namespace detail {

/**
 * @brief Function object to check if an index is within the bounds [begin, end).
 */
template <typename map_type>
struct bounds_checker {
  size_type begin;
  size_type end;

  __device__ bounds_checker(size_type begin_, size_type end_) : begin{begin_}, end{end_} {}

  __device__ bool operator()(map_type const index) { return ((index >= begin) && (index < end)); }
};

/**
 * @brief The operation to perform when a gather map index is out of bounds
 */
enum class gather_bitmask_op {
  DONT_CHECK,   // Don't check for out of bounds indices
  PASSTHROUGH,  // Preserve mask at rows with out of bounds indices
  NULLIFY,      // Nullify rows with out of bounds indices
};

template <gather_bitmask_op Op, typename MapIterator>
struct gather_bitmask_functor {
  table_device_view input;
  bitmask_type** masks;
  MapIterator gather_map;

  __device__ bool operator()(size_type mask_idx, size_type bit_idx)
  {
    auto row_idx = gather_map[bit_idx];
    auto col     = input.column(mask_idx);

    if (Op != gather_bitmask_op::DONT_CHECK) {
      bool out_of_range = is_signed_iterator<MapIterator>() ? (row_idx < 0 || row_idx >= col.size())
                                                            : row_idx >= col.size();
      if (out_of_range) {
        if (Op == gather_bitmask_op::PASSTHROUGH) {
          return bit_is_set(masks[mask_idx], bit_idx);
        } else if (Op == gather_bitmask_op::NULLIFY) {
          return false;
        }
      }
    }

    return col.is_valid(row_idx);
  }
};

/**
 * @brief Function for calling gather using iterators.
 *
 * Used by column_gatherer_impl definitions below.
 *
 * @tparam InputIterator Type for gather source data
 * @tparam OutputIterator Type for gather results
 * @tparam MapIterator Iterator type for the gather map
 *
 * @param source_itr Source data up to `source_size`
 * @param source_size Maximum index value for source data
 * @param target_itr Output iterator for gather result
 * @param gather_map_begin Start of the gather map
 * @param gather_map_end End of the gather map
 * @param nullify_out_of_bounds True if map values are checked against `source_size`
 * @param stream CUDA stream used for kernel launches.
 */
template <typename InputItr, typename OutputItr, typename MapIterator>
void gather_helper(InputItr source_itr,
                   size_type source_size,
                   OutputItr target_itr,
                   MapIterator gather_map_begin,
                   MapIterator gather_map_end,
                   bool nullify_out_of_bounds,
                   rmm::cuda_stream_view stream)
{
  using map_type = typename std::iterator_traits<MapIterator>::value_type;
  if (nullify_out_of_bounds) {
    thrust::gather_if(rmm::exec_policy(stream),
                      gather_map_begin,
                      gather_map_end,
                      gather_map_begin,
                      source_itr,
                      target_itr,
                      bounds_checker<map_type>{0, source_size});
  } else {
    thrust::gather(
      rmm::exec_policy(stream), gather_map_begin, gather_map_end, source_itr, target_itr);
  }
}

/**
 * @brief Function object for gathering a type-erased column.
 *
 * To be used with column_gatherer to provide specialization to handle
 * fixed-width, string and other types.
 *
 * @tparam Element Dispatched type for the column being gathered
 * @tparam MapIterator Iterator type for the gather map
 */
template <typename Element, typename MapIterator>
struct column_gatherer_impl {
  /**
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * This handles fixed width type column_views only.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   * map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapIterator gather_map_begin,
                                     MapIterator gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto const num_rows = cudf::distance(gather_map_begin, gather_map_end);
    auto const policy   = cudf::mask_allocation_policy::NEVER;
    auto destination_column =
      cudf::detail::allocate_like(source_column, num_rows, policy, stream, mr);

    using Type = device_storage_type_t<Element>;

    gather_helper(source_column.data<Type>(),
                  source_column.size(),
                  destination_column->mutable_view().template begin<Type>(),
                  gather_map_begin,
                  gather_map_end,
                  nullify_out_of_bounds,
                  stream);

    return destination_column;
  }
};

/**
 * @brief Function object for gathering a type-erased
 * column. To be used with column_gatherer to provide specialization for
 * string_view.
 *
 * @tparam MapIterator Iterator type for the gather map
 */
template <typename MapItType>
struct column_gatherer_impl<string_view, MapItType> {
  /**
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`. This handles string_view type column_views only.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   * map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (true == nullify_out_of_bounds) {
      return cudf::strings::detail::gather<true>(
        strings_column_view(source_column), gather_map_begin, gather_map_end, stream, mr);
    } else {
      return cudf::strings::detail::gather<false>(
        strings_column_view(source_column), gather_map_begin, gather_map_end, stream, mr);
    }
  }
};

/**
 * @brief Column gather specialization for list_view column type.
 *
 * @tparam MapItRoot Iterator type to access the incoming root column.
 *
 * This functor is invoked only on the root column of a hierarchy of list
 * columns. Recursion is handled internally.
 */
template <typename MapItRoot>
struct column_gatherer_impl<list_view, MapItRoot> {
  /**
   * @brief Gather a list column from a hierarchy of list columns.
   *
   * This function is similar to gather_list_nested() but the difference is
   * significant.  This particular level takes a templated gather map iterator of
   * any type.  As we start recursing, we need to be able to generate new gather
   * maps for each level.  To do this requires manifesting a buffer of intermediate
   * data. If we were to do that at level N and then wrap it in an anonymous iterator
   * to be passed to level N+1, these buffers of data would remain resident for the
   * entirety of the recursion.  But if level N+1 could create it's own iterator
   * internally from a buffer passed to it by level N, it could then -delete- that
   * buffer of data after using it, keeping the amount of extra memory needed
   * to a minimum. see comment on "memory optimization" inside cudf::list::gather_list_nested
   *
   * The tree of calls can be visualized like this:
   *
   * @code{.pseudo}
   * R :  this operator
   * N :  lists::detail::gather_list_nested
   * L :  lists::detail::gather_list_leaf
   *
   *        R
   *       / \
   *      L   N
   *           \
   *            N
   *             \
   *              ...
   *               \
   *                L
   * @endcode
   *
   * This is the start of the recursion - we will only ever get in here once.
   * We will only ever travel down the left branch or the right branch, and we
   * will always end up in a final call to gather_list_leaf.
   *
   * @param column View into the column to gather from
   * @param gather_map_begin iterator representing the start of the range to gather from
   * @param gather_map_end iterator representing the end of the range to gather from
   * @param nullify_out_of_bounds Nullify values in the gather map that are out of bounds
   * @param stream CUDA stream on which to execute kernels
   * @param mr Memory resource to use for all allocations
   *
   * @returns column with elements gathered based on the gather map
   *
   */
  std::unique_ptr<column> operator()(column_view const& column,
                                     MapItRoot gather_map_begin,
                                     MapItRoot gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    lists_column_view list(column);
    auto gather_map_size = std::distance(gather_map_begin, gather_map_end);
    // if the gather map is empty, return an empty column
    if (gather_map_size == 0) { return empty_like(column); }

    // generate gather_data for the next level (N+1)
    lists::detail::gather_data gd = nullify_out_of_bounds
                                      ? lists::detail::make_gather_data<true>(
                                          column, gather_map_begin, gather_map_size, stream, mr)
                                      : lists::detail::make_gather_data<false>(
                                          column, gather_map_begin, gather_map_size, stream, mr);

    // the nesting case.
    if (list.child().type() == cudf::data_type{type_id::LIST}) {
      // gather children
      auto child = lists::detail::gather_list_nested(list.get_sliced_child(stream), gd, stream, mr);

      // return the final column
      return make_lists_column(gather_map_size,
                               std::move(gd.offsets),
                               std::move(child),
                               0,
                               rmm::device_buffer{0, stream, mr});
    }

    // it's a leaf.  do a regular gather
    auto child = lists::detail::gather_list_leaf(list.get_sliced_child(stream), gd, stream, mr);

    // assemble final column
    return make_lists_column(gather_map_size,
                             std::move(gd.offsets),
                             std::move(child),
                             0,
                             rmm::device_buffer{0, stream, mr});
  }
};

/**
 * @brief Function object for gathering a type-erased
 * column. To be used with the cudf::type_dispatcher.
 */
struct column_gatherer {
  /**
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam Element Dispatched type for the column being gathered
   * @tparam MapIterator Iterator type for the gather map
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   * map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  template <typename Element, typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapIterator gather_map_begin,
                                     MapIterator gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    column_gatherer_impl<Element, MapIterator> gatherer{};

    return gatherer(
      source_column, gather_map_begin, gather_map_end, nullify_out_of_bounds, stream, mr);
  }
};

/**
 * @brief Column gather specialization for dictionary column type.
 */
template <typename MapItType>
struct column_gatherer_impl<dictionary32, MapItType> {
  /**
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   * map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return New dictionary column with gathered rows.
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    dictionary_column_view dictionary(source_column);
    auto output_count = std::distance(gather_map_begin, gather_map_end);
    if (output_count == 0) return make_empty_column(data_type{type_id::DICTIONARY32});
    // The gather could cause some keys to be abandoned -- no indices point to them.
    // In this case, we could do further work to remove the abandoned keys and
    // reshuffle the indices values.
    // We decided we will copy the keys for gather since the keys column should
    // be relatively smallish.
    // Also, there are scenarios where the keys are common with other dictionaries
    // and the original intention was to share the keys here.
    auto keys_copy = std::make_unique<column>(dictionary.keys(), stream, mr);
    // Perform gather on just the indices
    column_view indices = dictionary.get_indices_annotated();
    auto new_indices    = cudf::detail::allocate_like(
      indices, output_count, cudf::mask_allocation_policy::NEVER, stream, mr);
    gather_helper(
      cudf::detail::indexalator_factory::make_input_iterator(indices),
      indices.size(),
      cudf::detail::indexalator_factory::make_output_iterator(new_indices->mutable_view()),
      gather_map_begin,
      gather_map_end,
      nullify_out_of_bounds,
      stream);
    // dissect the column's contents
    auto indices_type = new_indices->type();
    auto null_count   = new_indices->null_count();  // get this before calling release()
    auto contents     = new_indices->release();     // new_indices will now be empty
    // build the output indices column from the contents' data component
    auto indices_column = std::make_unique<column>(indices_type,
                                                   static_cast<size_type>(output_count),
                                                   std::move(*(contents.data.release())),
                                                   rmm::device_buffer{0, stream, mr},
                                                   0);  // set null count to 0
    // finally, build the dictionary with the null_mask component and the keys and indices
    return make_dictionary_column(std::move(keys_copy),
                                  std::move(indices_column),
                                  std::move(*(contents.null_mask.release())),
                                  null_count);
  }
};

/**
 * @brief Function object for applying a transformation on the gathermap
 * that converts negative indices to positive indices
 *
 * A negative index `i` is transformed to `i + size`, where `size` is
 * the number of elements in the column being gathered from.
 * Allowable values for the index `i` are in the range `[-size, size)`.
 * Thus, when gathering from a column of size `10`, the index `-1`
 * is transformed to `9` (i.e., the last element), `-2` is transformed
 * to `8` (the second-to-last element) and so on.
 * Positive indices are unchanged by this transformation.
 */
template <typename map_type>
struct index_converter : public thrust::unary_function<map_type, map_type> {
  index_converter(size_type n_rows) : n_rows(n_rows) {}

  __device__ map_type operator()(map_type in) const { return ((in % n_rows) + n_rows) % n_rows; }
  size_type n_rows;
};

template <gather_bitmask_op Op, typename GatherMap>
void gather_bitmask(table_device_view input,
                    GatherMap gather_map_begin,
                    bitmask_type** masks,
                    size_type mask_count,
                    size_type mask_size,
                    size_type* valid_counts,
                    rmm::cuda_stream_view stream)
{
  if (mask_size == 0) { return; }

  constexpr size_type block_size = 256;
  using Selector                 = gather_bitmask_functor<Op, decltype(gather_map_begin)>;
  auto selector                  = Selector{input, masks, gather_map_begin};
  auto counting_it               = thrust::make_counting_iterator(0);
  auto kernel =
    valid_if_n_kernel<decltype(counting_it), decltype(counting_it), Selector, block_size>;

  cudf::detail::grid_1d grid{mask_size, block_size, 1};
  kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    counting_it, counting_it, selector, masks, mask_count, mask_size, valid_counts);
}

template <typename MapIterator>
void gather_bitmask(table_view const& source,
                    MapIterator gather_map,
                    std::vector<std::unique_ptr<column>>& target,
                    gather_bitmask_op op,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr)
{
  if (target.empty()) { return; }

  // Validate that all target columns have the same size
  auto const target_rows = target.front()->size();
  CUDF_EXPECTS(std::all_of(target.begin(),
                           target.end(),
                           [target_rows](auto const& col) { return target_rows == col->size(); }),
               "Column size mismatch");

  // Create null mask if source is nullable but target is not
  for (size_t i = 0; i < target.size(); ++i) {
    if ((source.column(i).nullable() or op == gather_bitmask_op::NULLIFY) and
        not target[i]->nullable()) {
      auto const state =
        op == gather_bitmask_op::PASSTHROUGH ? mask_state::ALL_VALID : mask_state::UNINITIALIZED;
      auto mask = detail::create_null_mask(target[i]->size(), state, stream, mr);
      target[i]->set_null_mask(std::move(mask), 0);
    }
  }

  // Make device array of target bitmask pointers
  thrust::host_vector<bitmask_type*> target_masks(target.size());
  std::transform(target.begin(), target.end(), target_masks.begin(), [](auto const& col) {
    return col->mutable_view().null_mask();
  });
  rmm::device_vector<bitmask_type*> d_target_masks(target_masks);

  auto const masks         = d_target_masks.data().get();
  auto const device_source = table_device_view::create(source, stream);
  auto d_valid_counts      = rmm::device_vector<size_type>(target.size());

  // Dispatch operation enum to get implementation
  auto const impl = [op]() {
    switch (op) {
      case gather_bitmask_op::DONT_CHECK:
        return gather_bitmask<gather_bitmask_op::DONT_CHECK, MapIterator>;
      case gather_bitmask_op::PASSTHROUGH:
        return gather_bitmask<gather_bitmask_op::PASSTHROUGH, MapIterator>;
      case gather_bitmask_op::NULLIFY:
        return gather_bitmask<gather_bitmask_op::NULLIFY, MapIterator>;
      default: CUDF_FAIL("Invalid gather_bitmask_op");
    }
  }();
  impl(*device_source,
       gather_map,
       masks,
       target.size(),
       target_rows,
       d_valid_counts.data().get(),
       stream);

  // Copy the valid counts into each column
  auto const valid_counts = thrust::host_vector<size_type>(d_valid_counts);
  for (size_t i = 0; i < target.size(); ++i) {
    if (target[i]->nullable()) {
      auto const null_count = target_rows - valid_counts[i];
      target[i]->set_null_count(null_count);
    }
  }
}

template <typename MapItRoot>
struct column_gatherer_impl<struct_view, MapItRoot> {
  std::unique_ptr<column> operator()(column_view const& column,
                                     MapItRoot gather_map_begin,
                                     MapItRoot gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    structs_column_view structs_column(column);
    auto gather_map_size{std::distance(gather_map_begin, gather_map_end)};
    if (gather_map_size == 0) { return empty_like(column); }

    std::vector<std::unique_ptr<cudf::column>> output_struct_members;
    std::transform(structs_column.child_begin(),
                   structs_column.child_end(),
                   std::back_inserter(output_struct_members),
                   [&gather_map_begin, &gather_map_end, nullify_out_of_bounds, stream, mr](
                     cudf::column_view const& col) {
                     return cudf::type_dispatcher(col.type(),
                                                  column_gatherer{},
                                                  col,
                                                  gather_map_begin,
                                                  gather_map_end,
                                                  nullify_out_of_bounds,
                                                  stream,
                                                  mr);
                   });

    gather_bitmask(
      // Table view of struct column.
      cudf::table_view{
        std::vector<cudf::column_view>{structs_column.child_begin(), structs_column.child_end()}},
      gather_map_begin,
      output_struct_members,
      nullify_out_of_bounds ? gather_bitmask_op::NULLIFY : gather_bitmask_op::DONT_CHECK,
      stream,
      mr);

    return cudf::make_structs_column(
      gather_map_size,
      std::move(output_struct_members),
      0,
      rmm::device_buffer{0, stream, mr},  // Null mask will be fixed up in cudf::gather().
      stream,
      mr);
  }
};

/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * tparam MapIterator Iterator type for the gather map
 * @param[in] source_table View into the table containing the input columns whose rows will be
 * gathered
 * @param[in] gather_map_begin Beginning of iterator range of integer indices that map the rows in
 * the source columns to rows in the destination columns
 * @param[in] gather_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] bounds_policy Policy to apply to account for possible out-of-bound indices
 * `DONT_CHECK` skips all bound checking for gather map values. `NULLIFY` coerces rows that
 * corresponds to out-of-bound indices in the gather map to be null elements. Callers should
 * use `DONT_CHECK` when they are certain that the gather_map contains only valid indices for
 * better performance. In case there are out-of-bound indices in the gather map, the behavior
 * is undefined. Defaults to `DONT_CHECK`.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return cudf::table Result of the gather
 */
template <typename MapIterator>
std::unique_ptr<table> gather(
  table_view const& source_table,
  MapIterator gather_map_begin,
  MapIterator gather_map_end,
  out_of_bounds_policy bounds_policy  = out_of_bounds_policy::DONT_CHECK,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  std::vector<std::unique_ptr<column>> destination_columns;

  // TODO: Could be beneficial to use streams internally here

  for (auto const& source_column : source_table) {
    // The data gather for n columns will be put on the first n streams
    destination_columns.push_back(
      cudf::type_dispatcher(source_column.type(),
                            column_gatherer{},
                            source_column,
                            gather_map_begin,
                            gather_map_end,
                            bounds_policy == out_of_bounds_policy::NULLIFY,
                            stream,
                            mr));
  }

  gather_bitmask_op const op = bounds_policy == out_of_bounds_policy::NULLIFY
                                 ? gather_bitmask_op::NULLIFY
                                 : gather_bitmask_op::DONT_CHECK;

  gather_bitmask(source_table, gather_map_begin, destination_columns, op, stream, mr);

  return std::make_unique<table>(std::move(destination_columns));
}

}  // namespace detail
}  // namespace cudf
