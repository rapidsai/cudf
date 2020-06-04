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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform_scan.h>

#include <cub/cub.cuh>

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
 * @brief Function object for gathering a type-erased
 * column. To be used with column_gatherer to provide specialization to handle
 * fixed-width, string and other types.
 *
 * @tparam Element Dispatched type for the column being gathered
 * @tparam MapIterator Iterator type for the gather map
 */
template <typename Element, typename MapIterator>
struct column_gatherer_impl {
  /*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`. This handles fixed width type column_views only.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   *map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapIterator gather_map_begin,
                                     MapIterator gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    size_type num_destination_rows      = std::distance(gather_map_begin, gather_map_end);
    cudf::mask_allocation_policy policy = cudf::mask_allocation_policy::NEVER;
    std::unique_ptr<column> destination_column =
      cudf::detail::allocate_like(source_column, num_destination_rows, policy, mr, stream);
    Element const* source_data{source_column.data<Element>()};
    Element* destination_data{destination_column->mutable_view().data<Element>()};

    using map_type = typename std::iterator_traits<MapIterator>::value_type;

    if (nullify_out_of_bounds) {
      thrust::gather_if(rmm::exec_policy(stream)->on(stream),
                        gather_map_begin,
                        gather_map_end,
                        gather_map_begin,
                        source_data,
                        destination_data,
                        bounds_checker<map_type>{0, source_column.size()});
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream),
                     gather_map_begin,
                     gather_map_end,
                     source_data,
                     destination_data);
    }

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
  /*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`. This handles string_view type column_views only.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather
   *map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    if (true == nullify_out_of_bounds) {
      return cudf::strings::detail::gather<true>(
        strings_column_view(source_column), gather_map_begin, gather_map_end, mr, stream);
    } else {
      return cudf::strings::detail::gather<false>(
        strings_column_view(source_column), gather_map_begin, gather_map_end, mr, stream);
    }
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
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return New dictionary column with gathered rows.
   */
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    dictionary_column_view dictionary(source_column);
    auto output_count = std::distance(gather_map_begin, gather_map_end);
    if (output_count == 0) return make_empty_column(data_type{DICTIONARY32});
    // The gather could cause some keys to be abandoned -- no indices point to them.
    // In this case, we could do further work to remove the abandoned keys and
    // reshuffle the indices values.
    // We decided we will copy the keys for gather since the keys column should
    // be relatively smallish.
    // Also, there are scenarios where the keys are common with other dictionaries
    // and the original intention was to share the keys here.
    auto keys_copy = std::make_unique<column>(dictionary.keys(), stream, mr);
    // create view of the indices column combined with the null mask
    // in order to call gather on it
    column_view indices(data_type{INT32},
                        dictionary.size(),
                        dictionary.indices().data<int32_t>(),
                        dictionary.null_mask(),
                        dictionary.null_count(),
                        dictionary.offset());
    column_gatherer_impl<int32_t, MapItType> index_gatherer;
    auto new_indices =
      index_gatherer(indices, gather_map_begin, gather_map_end, nullify_out_of_bounds, mr, stream);
    // dissect the column's contents
    auto null_count = new_indices->null_count();  // get this before it goes away
    auto contents   = new_indices->release();     // new_indices will now be empty
    // build the output indices column from the contents' data component
    auto indices_column = std::make_unique<column>(data_type{INT32},
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
 * @brief Function object for gathering a type-erased column as the leaf
 * of a hierarchy of nested list columns. To be used with the cudf::type_dispatcher.
 */
struct leaf_column_gatherer {
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
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   */
  template <typename Element,
            typename MapItType,
            std::enable_if_t<!std::is_same<Element, cudf::list_view>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    column_gatherer_impl<Element, MapItType> gatherer{};
    return gatherer(
      source_column, gather_map_begin, gather_map_end, nullify_out_of_bounds, mr, stream);
  }

  // we should never be encountering a list_view as a leaf of a hierarchy. this specialization
  // exists to break template recursion during compilation.
  template <typename Element,
            typename MapItType,
            std::enable_if_t<std::is_same<Element, cudf::list_view>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("Invalid recursive invocation of list gather");
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
   * @brief Generates the data needed to create a `gather_map` for the next level of
   * recursion in a hierarchy of list columns.
   *
   * Gathering from a single level of a list column is similar to gathering from
   * a string column.  Each row represents a list bounded by offsets. Example:
   *
   * Level 0 : List<List<int>>
   *           Size : 3
   *           Offsets : [0, 2, 5, 10]
   *
   * This represents a column with 3 rows.
   * Row 0 has 2 elements (bounded by offsets 0,2)
   * Row 1 has 3 elements (bounded by offsets 2,5)
   * Row 2 has 5 eleemnts (bounded by offsets 5,10)
   *
   * If we wanted to gather rows 0 and 2 the offsets for our outgoing column
   * would be the compacted ranges (0,2) and (5,10). The level 1 column
   * then looks like
   *
   * Level 1 : List<int>
   *           Size : 2
   *           Offsets : [0, 2, 7]
   *
   * However, we need to then gather one level further, because at the bottom we have
   * a column of integers.  We cannot gather the elements in the ranges (0, 2) and (2, 7).
   * Instead, we have to gather elements in the ranges from the Level 0 column (0, 2) and (5, 10).
   * So our gather_map iterator will need to know these "base" offsets to index properly.
   * Specifically:
   *
   * Offsets        : [0, 2, 7]    The offsets for Level 1
   * Base Offsets   : [0, 5]       The corresponding base offsets from Level 0
   *
   * Using this we can create an iterator that generates the sequence which properly indexes the
   * final integer values we want to gather.
   *
   * [0, 1, 5, 6, 7, 8, 9]
   *
   * Thinking generally, this means that to produce a gather_map for level N+1, we need to use the
   * offsets from level N and the "base" offsets from level N-1. So we are always carrying along
   * one extra buffer of these "base" offsets which keeps our memory usage well controlled.
   *
   * A concrete example:
   *
   * "column of lists of lists of ints"
   * {
   *   {
   *      {2, 3}, {4, 5}
   *   },
   *   {
   *      {6, 7, 8}, {9, 10, 11}, {12, 13, 14}
   *   },
   *   {
   *      {15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}
   *   }
   * }
   *
   * List<List<int32_t>>:
   * Length : 3
   * Offsets : 0, 2, 5, 10
   * Children :
   *    List<int32_t>:
   *    Length : 10
   *    Offsets : 0, 2, 4, 7, 10, 13, 15, 17, 19, 21, 23
   *       Children :
   *           2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
   *
   * Final column, doing gather([0, 2])
   *
   * List<List<int32_t>>:
   * Length : 2
   * Offsets : 0, 2, 7
   * Children :
   *    List<int32_t>:
   *    Length : 7
   *    Offsets : 0, 2, 4, 6, 8, 10, 12, 14
   *       Children :
   *          2, 3, 4, 5, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
   *
   * @tparam MapItType Iterator type to access the incoming column.
   * @param source_column View into the column to gather from
   * @param gather_map Iterator access to the gather map for `source_column`
   * map
   * @param gather_map_size Size of the gather map.
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *
   * @returns std::pair containing two columns of data.  the compacted offsets
   *          for the child column and the base (original) offsets from the source
   *          column that matches each of those new offsets.
   *
   */
  template <typename MapItType>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_gather_offsets(
    lists_column_view const& source_column,
    MapItType gather_map,
    size_type gather_map_size,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    // size of the gather map is the # of output rows
    size_type output_count = gather_map_size;
    size_type offset_count = output_count + 1;

    // offsets of the source column
    size_type const* src_offsets{source_column.offsets().data<size_type>()};

    // outgoing offsets.  these will persist as output from the entire gather operation
    auto dst_offsets_c = cudf::make_fixed_width_column(
      data_type{INT32}, offset_count, mask_state::UNALLOCATED, stream, mr);
    mutable_column_view dst_offsets_v = dst_offsets_c->mutable_view();
    size_type* dst_offsets            = dst_offsets_v.data<size_type>();

    // generate the compacted outgoing offsets.
    auto count_iter = thrust::make_counting_iterator<size_type>(0);
    thrust::plus<size_type> sum;
    thrust::transform_exclusive_scan(
      rmm::exec_policy(stream)->on(stream),
      count_iter,
      count_iter + offset_count,
      dst_offsets,
      [gather_map, output_count, src_offsets] __device__(size_type index) -> size_type {
        // last offset index is always the previous offset_index + 1, since each entry in the gather
        // map represents a virtual pair of offsets
        size_type offset_index =
          index < output_count ? gather_map[index] : gather_map[index - 1] + 1;
        // the length of this list
        return src_offsets[offset_index + 1] - src_offsets[offset_index];
      },
      0,
      sum);

    // for each span of offsets (each output row) we need to know the original offset value to build
    // the gather map for the next level.  this data is temporary and will only persist until the
    // next level of recursion is done using it.  This way, even if we have a hierarchy 1000 levels
    // deep, we will only ever have at most 2 of these extra columns in memory.
    auto base_offsets = cudf::make_fixed_width_column(
      data_type{INT32}, output_count, mask_state::UNALLOCATED, stream, mr);
    mutable_column_view base_offsets_mcv = base_offsets->mutable_view();
    size_type* base_offsets_p            = base_offsets_mcv.data<size_type>();

    // generate the base offsets
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      gather_map,
                      gather_map + offset_count,
                      base_offsets_p,
                      [src_offsets] __device__(size_type index) { return src_offsets[index]; });

    return {std::move(dst_offsets_c), std::move(base_offsets)};
  }

  /**
   * @brief The information needed to create an iterator to gather level N+1
   *
   * See documentation for make_gather_offsets for a detailed explanation.
   */
  struct gather_data {
    size_type const* offsets;
    std::unique_ptr<column> base_offsets;
  };

/**
 * @brief Macro for creating a gather map iterator to gather from level N+1.
 *
 * The iterator for each level is unique and we can't use templates because the
 * operation is recursive.  There is a restriction in nvcc that prevents you
 * from returning a __device__ lambda from a function, so this cannot be implemented
 * as function.
 *
 * The iterator needed for gathering at level N+1 needs to reference the offsets
 * from level N and the "base" offsets used from level N-1.  An example of
 * the gather map needed for level N+1 (see documentation for make_gather_offsets for
 * the full example)
 *
 * level N-1 offsets               : [0, 2, 5, 10], gather map[0, 2]
 *
 * level N offsets                 : [0, 2, 7]
 * "base" offsets from level N-1   : [0, 5]
 *
 * desired output sequence from the level N+1 gather map
 * [0, 1, 5, 6, 7, 8, 9]
 *
 * The generation of this sequence in this iterator works as follows
 *
 * step 1, generate row index sequence
 * [0, 0, 1, 1, 1, 1, 1]
 * step 2, generate row subindex sequence
 * [0, 1, 0, 1, 2, 3, 4]
 * step 3, add base offsets to get the final sequence
 * [0, 1, 5, 6, 7, 8, 9]
 *
 * @param __name variable name of the resulting iterator
 * @param gather_data the `gather_data` struct needed to generate the map sequence
 *
 * @returns an iterator that produces the gather map for level N+1
 *
 */
#define LIST_GATHER_ITERATOR(__name, gather_data)                                                                      \
  auto __name##span_upper_bound = thrust::make_transform_iterator(                                                     \
    thrust::make_counting_iterator<size_type>(0),                                                                      \
    [offsets = gather_data.offsets] __device__(size_type index) { return offsets[index + 1]; });                       \
  column_view __name##base_offsets_v(*gather_data.base_offsets);                                                       \
  size_type const* __name##base_offsets = __name##base_offsets_v.data<size_type>();                                    \
  auto __name                           = thrust::make_transform_iterator(                                             \
    thrust::make_counting_iterator<size_type>(0),                                            \
    [span_upper_bound = __name##span_upper_bound,                                            \
     offset_count     = __name##base_offsets_v.size(),                                       \
     offsets          = gather_data.offsets,                                                 \
     base_offsets     = __name##base_offsets] __device__(size_type index) {                      \
      auto bound = thrust::upper_bound(                                                      \
        thrust::device, span_upper_bound, span_upper_bound + offset_count, index);           \
      size_type offset_index    = thrust::distance(span_upper_bound, bound);                 \
      size_type offset_subindex = offset_index == 0 ? index : index - offsets[offset_index]; \
      return offset_subindex + base_offsets[offset_index];                                   \
    });

  /**
   * @brief Gather a leaf column from a hierarchy of list columns. The recursion
   * terminates here.
   *
   * @param column View into the column to gather from
   * @param gd The gather_data needed to construct a gather map iterator for this level
   * @param nullify_out_of_bounds Nullify values in the gather map that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *
   * @returns column with elements gathered based on `gather_data`
   *
   */
  std::unique_ptr<column> gather_list_leaf(column_view const& column,
                                           gather_data const& gd,
                                           bool nullify_out_of_bounds,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    // gather map for our child
    LIST_GATHER_ITERATOR(child_gather_map, gd);

    // size of the child gather map
    size_type child_gather_map_size;
    CUDA_TRY(cudaMemcpy(&child_gather_map_size,
                        gd.offsets + gd.base_offsets->size(),
                        sizeof(size_type),
                        cudaMemcpyDeviceToHost));

    // otherwise, just call a regular gather
    return cudf::type_dispatcher(column.type(),
                                               leaf_column_gatherer{},
                                               column,
                                               child_gather_map,
                                               child_gather_map + child_gather_map_size,
                                               nullify_out_of_bounds,
                                               mr,
                                               stream);
  }

  /**
   * @brief Gather a list column from a hierarchy of list columns. The recursion
   * continues from here at least 1 level further
   *
   * @param list View into the list column to gather from
   * @param gd The gather_data needed to construct a gather map iterator for this level
   * @param nullify_out_of_bounds Nullify values in the gather map that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *
   * @returns column with elements gathered based on `gather_data`
   *
   */
  std::unique_ptr<column> gather_list_nested(lists_column_view const& list,
                                             gather_data& parent,
                                             bool nullify_out_of_bounds,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
  {
    // build the gather map iterator
    LIST_GATHER_ITERATOR(gather_map_begin, parent);

    // size of the child gather map
    size_type gather_map_size;
    CUDA_TRY(cudaMemcpy(&gather_map_size,
                        parent.offsets + parent.base_offsets->size(),
                        sizeof(size_type),
                        cudaMemcpyDeviceToHost));

    // generate gather_data for this level
    auto offset_result = make_gather_offsets(list, gather_map_begin, gather_map_size, mr, stream);
    column_view offsets_v(*offset_result.first);
    gather_data gd{offsets_v.data<size_type>(), std::move(offset_result.second)};

    // memory optimization. now that we have generated the base offset data we need for level N+1,
    // we are no longer going to be using gather_map_begin, so the base offset data it references
    // from level N-1 above us can be released
    parent.base_offsets.release();

    // the nesting case.  we have to recurse through the hierarchy, but we can't do that via
    // templates, so we can't pass an iterator.  so instead call a function that can build it's own
    // iterator that does the same thing as -this- function.
    if (list.child().type() == cudf::data_type{LIST}) {
      // gather children
      auto child = gather_list_nested(list.child(), gd, nullify_out_of_bounds, mr, stream);

      // return the final column
      return make_lists_column(
        gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
    }

    // it's a leaf.  do a regular gather
    auto child = gather_list_leaf(list.child(), gd, nullify_out_of_bounds, mr, stream);

    // assemble final column
    return make_lists_column(
      gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
  }

  /**
   * @brief Gather a list column from a hierarchy of list columns. This is the start
   * of the recursion - we will only ever get in here once.
   *
   * This function is almost identical to gather_list_nested() but this similarity
   * is a necessary evil to handle the recursive nature of the operation. This functor
   * is called in a templated way, but we can't continue doing that for the nesting.
   * The tree of calls can be visualized like this:
   *
   * R :  this operator
   * N :  gather_list_nested
   * L :  gather_list_leaf
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
   *
   * We will only ever travel down the left branch or the right branch
   *
   * @param column View into the column to gather from
   * @param gather_map_begin iterator representing the start of the range to gather from
   * @param gather_map_end iterator representing the end of the range to gather from
   * @param nullify_out_of_bounds Nullify values in the gather map that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *
   * @returns column with elements gathered based on the gather map
   *
   */
  std::unique_ptr<column> operator()(column_view const& column,
                                     MapItRoot gather_map_begin,
                                     MapItRoot gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    lists_column_view list(column);
    auto gather_map_size = std::distance(gather_map_begin, gather_map_end);
    if (gather_map_size == 0) { return make_empty_column(data_type{LIST}); }

    // generate gather_data for this level
    auto offset_result = make_gather_offsets(column, gather_map_begin, gather_map_size, mr, stream);
    column_view offsets_v(*offset_result.first);
    gather_data gd{offsets_v.data<size_type>(), std::move(offset_result.second)};

    // the nesting case.  we have to recurse through the hierarchy, but we can't do that via
    // templates, so we can't pass an iterator.  so instead call a function that can build it's own
    // iterator that does the same thing as -this- function.
    if (list.child().type() == cudf::data_type{LIST}) {
      // gather children
      auto child = gather_list_nested(list.child(), gd, nullify_out_of_bounds, mr, stream);

      // return the final column
      return make_lists_column(
        gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
    }

    // it's a leaf.  do a regular gather
    auto child = gather_list_leaf(list.child(), gd, nullify_out_of_bounds, mr, stream);

    // assemble final column
    return make_lists_column(
      gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
  }
};

/**
 * @brief Function object for gathering a type-erased
 * column. To be used with the cudf::type_dispatcher.
 *
 */
struct column_gatherer {
  /*
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
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  template <typename Element, typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapIterator gather_map_begin,
                                     MapIterator gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    column_gatherer_impl<Element, MapIterator> gatherer{};

    return gatherer(
      source_column, gather_map_begin, gather_map_end, nullify_out_of_bounds, mr, stream);
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
                    cudaStream_t stream)
{
  if (mask_size == 0) { return; }

  constexpr size_type block_size = 256;
  using Selector                 = gather_bitmask_functor<Op, decltype(gather_map_begin)>;
  auto selector                  = Selector{input, masks, gather_map_begin};
  auto counting_it               = thrust::make_counting_iterator(0);
  auto kernel =
    valid_if_n_kernel<decltype(counting_it), decltype(counting_it), Selector, block_size>;

  cudf::detail::grid_1d grid{mask_size, block_size, 1};
  kernel<<<grid.num_blocks, block_size, 0, stream>>>(
    counting_it, counting_it, selector, masks, mask_count, mask_size, valid_counts);
}

template <typename MapIterator>
void gather_bitmask(table_view const& source,
                    MapIterator gather_map,
                    std::vector<std::unique_ptr<column>>& target,
                    gather_bitmask_op op,
                    rmm::mr::device_memory_resource* mr,
                    cudaStream_t stream)
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
      auto mask = create_null_mask(target[i]->size(), state, stream, mr);
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

template <typename MapIterator>
std::unique_ptr<column> gather(
  column_view const& source_column,
  MapIterator gather_map_begin,
  MapIterator gather_map_end,
  bool nullify_out_of_bounds          = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  return cudf::type_dispatcher(source_column.type(),
                                             column_gatherer{},
                                             source_column,
                                             gather_map_begin,
                                             gather_map_end,
                                             nullify_out_of_bounds,
                                             mr,
                                             stream);
}

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
 * @param[in] nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return cudf::table Result of the gather
 */
template <typename MapIterator>
std::unique_ptr<table> gather(table_view const& source_table,
                              MapIterator gather_map_begin,
                              MapIterator gather_map_end,
                              bool nullify_out_of_bounds          = false,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream                 = 0)
{
  std::vector<std::unique_ptr<column>> destination_columns;

  // TODO: Could be beneficial to use streams internally here

  for (auto const& source_column : source_table) {
    // The data gather for n columns will be put on the first n streams
    destination_columns.push_back(cudf::type_dispatcher(source_column.type(),
                                                        column_gatherer{},
                                                        source_column,
                                                        gather_map_begin,
                                                        gather_map_end,
                                                        nullify_out_of_bounds,
                                                        mr,
                                                        stream));
  }

  auto const op =
    nullify_out_of_bounds ? gather_bitmask_op::NULLIFY : gather_bitmask_op::DONT_CHECK;
  gather_bitmask(source_table, gather_map_begin, destination_columns, op, mr, stream);

  return std::make_unique<table>(std::move(destination_columns));
}

}  // namespace detail
}  // namespace cudf
