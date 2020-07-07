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
#include <thrust/binary_search.h>
#include <thrust/iterator/iterator_facade.h>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/detail/gather.cuh>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief List gatherer function object.
 *
 * The iterator needed for gathering at level N+1 needs to reference the offsets
 * from level N and the "base" offsets used from level N-1.  An example of
 * the gather map needed for level N+1 (see documentation for make_gather_data for
 * the full example)
 *
 * @code{.pseudo}
 * level N-1 offsets               : [0, 2, 5, 10], gather map[0, 2]
 *
 * level N offsets                 : [0, 2, 7]
 * "base" offsets from level N-1   : [0, 5]
 *
 * desired output sequence for the level N+1 gather map
 * [0, 1, 5, 6, 7, 8, 9]
 *
 * The generation of this sequence in this functor works as follows
 *
 * step 1, generate row index sequence
 * [0, 0, 1, 1, 1, 1, 1]
 * step 2, generate row subindex sequence
 * [0, 1, 0, 1, 2, 3, 4]
 * step 3, add base offsets to get the final sequence
 * [0, 1, 5, 6, 7, 8, 9]
 * @endcode
 *
 */
struct list_gatherer {
  typedef size_type argument_type;
  typedef size_type result_type;

  size_t offset_count;
  size_type const* base_offsets;
  size_type const* offsets;

  list_gatherer(gather_data const& gd)
    : offset_count{gd.base_offsets.size()},
      base_offsets{gd.base_offsets.data()},
      offsets{gd.offsets->mutable_view().data<size_type>()}
  {
  }

  __device__ result_type operator()(argument_type index)
  {
    // the "upper bound" of the span for a given offset is always offsets+1;
    size_type const* upper_bound_start = offsets + 1;
    // "step 1" from above
    auto const bound =
      thrust::upper_bound(thrust::seq, upper_bound_start, upper_bound_start + offset_count, index);
    size_type offset_index = thrust::distance(upper_bound_start, bound);
    // "step 2" from above
    size_type offset_subindex = offset_index == 0 ? index : index - offsets[offset_index];
    // "step 3" from above
    return offset_subindex + base_offsets[offset_index];
  }
};

/**
 * @copydoc cudf::lists::detail::gather_list_leaf
 *
 */
std::unique_ptr<column> gather_list_leaf(column_view const& column,
                                         gather_data const& gd,
                                         cudaStream_t stream,
                                         rmm::mr::device_memory_resource* mr)
{
  // gather map iterator for this level (N)
  auto gather_map_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), list_gatherer{gd});
  size_type gather_map_size = gd.gather_map_size;

  // call the normal gather
  auto leaf_column =
    cudf::type_dispatcher(column.type(),
                          cudf::detail::column_gatherer{},
                          column,
                          gather_map_begin,
                          gather_map_begin + gather_map_size,
                          // note : we don't need to bother checking for out-of-bounds here since
                          // our inputs at this stage aren't coming from the user.
                          false,
                          stream,
                          mr);

  // the column_gatherer doesn't create the null mask because it expects
  // that will be done in the gather_bitmask() step.  however, gather_bitmask()
  // only happens at the root level, and by definition this column is a
  // leaf.  so we have to generate the bitmask ourselves.
  // TODO : it might make sense to expose a gather() function that takes a column_view and
  // returns a column that does this work correctly.
  size_type null_count = column.null_count();
  if (null_count > 0) {
    auto list_cdv = column_device_view::create(column);
    auto validity = cudf::detail::valid_if(
      gather_map_begin,
      gather_map_begin + gd.gather_map_size,
      [cdv = *list_cdv] __device__(int index) { return cdv.is_valid(index) ? true : false; },
      stream,
      mr);

    leaf_column->set_null_mask(std::move(validity.first), validity.second);
  }

  return std::move(leaf_column);
}

/**
 * @copydoc cudf::lists::detail::gather_list_nested
 *
 */
std::unique_ptr<column> gather_list_nested(cudf::lists_column_view const& list,
                                           gather_data& gd,
                                           cudaStream_t stream,
                                           rmm::mr::device_memory_resource* mr)
{
  // gather map iterator for this level (N)
  auto gather_map_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), list_gatherer{gd});
  size_type gather_map_size = gd.gather_map_size;

  // gather the bitmask, if relevant
  rmm::device_buffer null_mask{0, stream, mr};
  size_type null_count = list.null_count();
  if (null_count > 0) {
    auto list_cdv = column_device_view::create(list.parent());
    auto validity = cudf::detail::valid_if(
      gather_map_begin,
      gather_map_begin + gather_map_size,
      [cdv = *list_cdv] __device__(int index) { return cdv.is_valid(index) ? true : false; },
      stream,
      mr);
    null_mask  = std::move(validity.first);
    null_count = validity.second;
  }

  // generate gather_data for next level (N+1), potentially recycling the temporary
  // base_offsets buffer.
  gather_data child_gd = make_gather_data<false>(
    list, gather_map_begin, gather_map_size, stream, mr, std::move(gd.base_offsets));

  // the nesting case.
  if (list.child().type() == cudf::data_type{type_id::LIST}) {
    // gather children.
    auto child = gather_list_nested(list.child(), child_gd, stream, mr);

    // return the nested column
    return make_lists_column(gather_map_size,
                             std::move(child_gd.offsets),
                             std::move(child),
                             null_count,
                             std::move(null_mask));
  }

  // it's a leaf.  do a regular gather
  auto child = gather_list_leaf(list.child(), child_gd, stream, mr);

  // assemble final column
  return make_lists_column(gather_map_size,
                           std::move(child_gd.offsets),
                           std::move(child),
                           null_count,
                           std::move(null_mask));
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
