#pragma once

#include <thrust/transform_scan.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>

namespace cudf {
namespace lists {
namespace detail {

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
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns std::pair containing two columns of data.  the compacted offsets
 *          for the child column and the base (original) offsets from the source
 *          column that matches each of those new offsets.
 *
 */
template <bool NullifyOutOfBounds, typename MapItType>
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_gather_offsets(
  cudf::lists_column_view const& source_column,
  MapItType gather_map,
  size_type gather_map_size,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  // size of the gather map is the # of output rows
  size_type output_count = gather_map_size;
  size_type offset_count = output_count + 1;

  // offsets of the source column
  size_type const* src_offsets{source_column.offsets().data<size_type>()};
  int src_size = source_column.size();

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
    [gather_map, output_count, src_offsets, src_size] __device__(size_type index) -> size_type {
      // last offset index is always the previous offset_index + 1, since each entry in the gather
      // map represents a virtual pair of offsets
      size_type offset_index = index < output_count ? gather_map[index] : gather_map[index - 1] + 1;

      // if this is an invalid index, this will be a NULL list
      if (NullifyOutOfBounds && ((offset_index < 0) || (offset_index >= src_size))) { return 0; }

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
                    [src_offsets, output_count, src_size] __device__(size_type index) {
                      // if this is an invalid index, this will be a NULL list
                      if (NullifyOutOfBounds && ((index < 0) || (index >= src_size))) { return 0; }
                      return src_offsets[index];
                    });

  return {std::move(dst_offsets_c), std::move(base_offsets)};
}

/**
 * @brief Gather a list column from a hierarchy of list columns. The recursion
 * continues from here at least 1 level further
 *
 * @param list View into the list column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 *
 */
std::unique_ptr<column> gather_list_nested(
  lists_column_view const& list,
  gather_data& parent,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Gather a leaf column from a hierarchy of list columns. The recursion
 * terminates here.
 *
 * @param column View into the column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 *
 */
std::unique_ptr<column> gather_list_leaf(
  column_view const& column,
  gather_data const& gd,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace lists
}  // namespace cudf