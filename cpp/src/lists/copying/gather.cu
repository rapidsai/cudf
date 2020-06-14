
#include <thrust/binary_search.h>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/detail/gather.cuh>

namespace cudf {
namespace lists {
namespace detail {

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
        thrust::seq, span_upper_bound, span_upper_bound + offset_count, index);              \
      size_type offset_index    = thrust::distance(span_upper_bound, bound);                 \
      size_type offset_subindex = offset_index == 0 ? index : index - offsets[offset_index]; \
      return offset_subindex + base_offsets[offset_index];                                   \
    });

/**
 * @copydoc cudf::lists::detail::gather_list_leaf
 *
 */
std::unique_ptr<column> gather_list_leaf(column_view const& column,
                                         gather_data const& gd,
                                         bool nullify_out_of_bounds,
                                         cudaStream_t stream,
                                         rmm::mr::device_memory_resource* mr)
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
                               cudf::detail::column_gatherer{},
                               column,
                               child_gather_map,
                               child_gather_map + child_gather_map_size,
                               nullify_out_of_bounds,
                               stream,
                               mr);
}

/**
 * @copydoc cudf::lists::detail::gather_list_nested
 *
 */
std::unique_ptr<column> gather_list_nested(cudf::lists_column_view const& list,
                                           gather_data& parent,
                                           bool nullify_out_of_bounds,
                                           cudaStream_t stream,
                                           rmm::mr::device_memory_resource* mr)
{
  LIST_GATHER_ITERATOR(gather_map_begin, parent);

  // size of the child gather map
  size_type gather_map_size = 0;
  CUDA_TRY(cudaMemcpy(&gather_map_size,
                      parent.offsets + parent.base_offsets->size(),
                      sizeof(size_type),
                      cudaMemcpyDeviceToHost));

  // generate gather_data for this level
  auto offset_result =
    nullify_out_of_bounds
      ? make_gather_offsets<true>(list, gather_map_begin, gather_map_size, stream, mr)
      : make_gather_offsets<false>(list, gather_map_begin, gather_map_size, stream, mr);
  column_view offsets_v(*offset_result.first);
  gather_data gd{offsets_v.data<size_type>(), std::move(offset_result.second)};

  // memory optimization. now that we have generated the base offset data we need for level N+1,
  // we are no longer going to be using gather_map_begin, so the base offset data it references
  // from level N-1 above us can be released
  parent.base_offsets.release();

  // the nesting case.  we have to recurse through the hierarchy, but we can't do that via
  // templates, so we can't pass an iterator.  so we will pass the data needed to create
  // the necessary iterator, and the functions will build the iterator themselves.
  if (list.child().type() == cudf::data_type{LIST}) {
    // gather children.
    // note : we don't need to bother checking for out-of-bounds here since
    // our inputs at this stage aren't coming from the user.
    auto child = gather_list_nested(list.child(), gd, false, stream, mr);

    // return the nested column
    return make_lists_column(
      gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
  }

  // it's a leaf.  do a regular gather
  // note : we don't need to bother checking for out-of-bounds here since
  // our inputs at this stage aren't coming from the user.
  auto child = gather_list_leaf(list.child(), gd, false, stream, mr);

  // assemble final column
  return make_lists_column(
    gather_map_size, std::move(offset_result.first), std::move(child), 0, rmm::device_buffer{});
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf