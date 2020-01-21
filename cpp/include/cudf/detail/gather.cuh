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

#include <cudf/types.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/detail/valid_if.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>

namespace cudf {
namespace experimental {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Function object to check if an index is within the bounds [begin,
 * end).
 *
 *---------------------------------------------------------------------------**/
template <typename map_type>
struct bounds_checker {
  size_type begin;
  size_type end;

  __device__ bounds_checker(size_type begin_, size_type end_)
    : begin{begin_}, end{end_} {}

  __device__ bool operator()(map_type const index) {
    return ((index >= begin) && (index < end));
  }
};

template <bool ignore_out_of_bounds, typename MapIterator>
struct gather_bitmask_functor {
  table_device_view input;
  bitmask_type** masks;
  MapIterator gather_map;

  __device__ bool operator()(size_type mask_idx, size_type bit_idx) {
    auto row_idx = gather_map[bit_idx];
    auto col = input.column(mask_idx);

    if (ignore_out_of_bounds) {
      if (row_idx < 0 || row_idx >= col.size()) {
        return bit_is_set(masks[mask_idx], bit_idx);
      }
    }

    return col.is_valid(row_idx);
  }
};

/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * column. To be used with column_gatherer to provide specialization to handle
 * fixed-width, string and other types.
 *
 * @tparam Element Dispatched type for the column being gathered
 * @tparam MapIterator Iterator type for the gather map
 *---------------------------------------------------------------------------**/
template<typename Element, typename MapIterator>
struct column_gatherer_impl
{
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`. This handles fixed width type column_views only.
   *
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
    std::unique_ptr<column> operator()(column_view const& source_column,
                                       MapIterator gather_map_begin,
                                       MapIterator gather_map_end,
                                       bool nullify_out_of_bounds,
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream) {

      size_type num_destination_rows = std::distance(gather_map_begin, gather_map_end);
      cudf::experimental::mask_allocation_policy policy =
        cudf::experimental::mask_allocation_policy::RETAIN;
      if (nullify_out_of_bounds) {
        policy = cudf::experimental::mask_allocation_policy::ALWAYS;
      }
      std::unique_ptr<column> destination_column =
          cudf::experimental::detail::allocate_like(source_column, num_destination_rows,
                          policy, mr, stream);
      Element const *source_data{source_column.data<Element>()};
      Element *destination_data{destination_column->mutable_view().data<Element>()};

      using map_type = typename std::iterator_traits<MapIterator>::value_type;

      if (nullify_out_of_bounds) {
        CUDA_TRY(cudaMemsetAsync(
              destination_column->mutable_view().null_mask(),
              0,
              bitmask_allocation_size_bytes(destination_column->size()),
              stream));

        thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map_begin,
                          gather_map_end, gather_map_begin,
                          source_data, destination_data,
                          bounds_checker<map_type>{0, source_column.size()});
      } else {
        thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map_begin,
                       gather_map_end, source_data, destination_data);
      }

      return destination_column;
    }
  };

/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * column. To be used with column_gatherer to provide specialization for
 * string_view.
 *
 * @tparam MapIterator Iterator type for the gather map
 *---------------------------------------------------------------------------**/

template<typename MapItType>
struct column_gatherer_impl<string_view, MapItType>
{
 /**---------------------------------------------------------------------------*
  * @brief Type-dispatched function to gather from one column to another based
  * on a `gather_map`. This handles string_view type column_views only.
  *
  * @param source_column View into the column to gather from
  * @param gather_map_begin Beginning of iterator range of integral values representing the gather map
  * @param gather_map_end End of iterator range of integral values representing the gather map
  * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
  * @param mr Memory resource to use for all allocations
  * @param stream CUDA stream on which to execute kernels
  *---------------------------------------------------------------------------**/
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapItType gather_map_begin,
                                     MapItType gather_map_end,
                                     bool nullify_out_of_bounds,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {
      if (true == nullify_out_of_bounds) {
        return cudf::strings::detail::gather<true>(
                       strings_column_view(source_column),
                       gather_map_begin, gather_map_end,
                       mr, stream);
      } else {
        return cudf::strings::detail::gather<false>(
                       strings_column_view(source_column),
                       gather_map_begin, gather_map_end,
                       mr, stream);
      }
  }

};

/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * column. To be used with the cudf::type_dispatcher.
 *
 *---------------------------------------------------------------------------**/
struct column_gatherer
{
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam Element Dispatched type for the column being gathered
   * @tparam MapIterator Iterator type for the gather map
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename Element, typename MapIterator>
    std::unique_ptr<column> operator()(column_view const& source_column,
                                       MapIterator gather_map_begin,
                                       MapIterator gather_map_end,
                                       bool nullify_out_of_bounds,
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream) {
      column_gatherer_impl<Element, MapIterator> gatherer{};

      return gatherer(source_column, gather_map_begin,
                    gather_map_end, nullify_out_of_bounds,
                    mr, stream);
  }
};

/**---------------------------------------------------------------------------*
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
*---------------------------------------------------------------------------**/
template <typename map_type>
struct index_converter : public thrust::unary_function<map_type,map_type>
{
  index_converter(size_type n_rows)
  : n_rows(n_rows) {}

  __device__
  map_type operator()(map_type in) const
  {
    return ((in % n_rows) + n_rows) % n_rows;
  }
  size_type n_rows;
};

template<bool ignore_out_of_bounds, typename GatherMap>
void gather_bitmask(table_device_view input,
                    GatherMap gather_map_begin,
                    bitmask_type** masks,
                    size_type mask_count,
                    size_type mask_size,
                    size_type* valid_counts,
                    cudaStream_t stream)
{
  if (mask_size == 0) {
    return;
  }

  constexpr size_type block_size = 256;
  using Selector = gather_bitmask_functor<ignore_out_of_bounds, decltype(gather_map_begin)>;
  auto selector = Selector{ input, masks, gather_map_begin };
  auto counting_it = thrust::make_counting_iterator(0);
  auto kernel = valid_if_n_kernel<decltype(counting_it), decltype(counting_it), Selector, block_size>;

  cudf::experimental::detail::grid_1d grid { mask_size, block_size, 1 };
  kernel<<<grid.num_blocks, block_size, 0, stream>>>(counting_it,
                                                     counting_it,
                                                     selector,
                                                     masks,
                                                     mask_count,
                                                     mask_size,
                                                     valid_counts);
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
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the source table. If `check_bounds == false`, the behavior is undefined.
 *
 * tparam MapIterator Iterator type for the gather map
 * @param[in] source_table View into the table containing the input columns whose rows will be gathered
 * @param[in] gather_map_begin Beginning of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] gather_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] check_bounds Optionally perform bounds checking on the values of `gather_map` and throw
 * an error if any of its values are out of bounds.
 * @param[in] nullify_out_of_bounds Nullify values in `gather_map` that are out of bounds. Currently
 * incompatible with `allow_negative_indices`, i.e., setting both to `true` is undefined.
 * @param[in] allow_negative_indices Interpret each negative index `i` in the gathermap as the
 * positive index `i+num_source_rows`.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The CUDA stream on which to execute kernels
 * @return cudf::table Result of the gather
 */
template <typename MapIterator>
std::unique_ptr<table>
gather(table_view const& source_table, MapIterator gather_map_begin,
       MapIterator gather_map_end, bool check_bounds = false,
       bool nullify_out_of_bounds = false,
       bool allow_negative_indices = false,
       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
       cudaStream_t stream = 0) {
  auto num_destination_rows = std::distance(gather_map_begin, gather_map_end);

  std::vector<std::unique_ptr<column>> destination_columns;

  // TODO: Could be beneficial to use streams internally here

  for(auto const& source_column : source_table) {
    // The data gather for n columns will be put on the first n streams
    destination_columns.push_back(
                                  cudf::experimental::type_dispatcher(source_column.type(),
                                                                      column_gatherer{},
                                                                      source_column,
                                                                      gather_map_begin,
                                                                      gather_map_end,
                                                                      nullify_out_of_bounds,
                                                                      mr,
                                                                      stream));

  }

  std::unique_ptr<table> destination_table = std::make_unique<table>(std::move(destination_columns));

  rmm::device_vector<cudf::size_type> valid_counts(source_table.num_columns(), 0);

  auto source_table_view = table_device_view::create(source_table);
  std::vector<bitmask_type*> host_masks(destination_table->num_columns());
  auto mutable_destination_table = destination_table->mutable_view();
  std::transform(mutable_destination_table.begin(), mutable_destination_table.end(),
                    host_masks.begin(), [] (auto col){
                        return  col.nullable()?col.null_mask():nullptr;
                    });

  rmm::device_vector<bitmask_type*> masks(host_masks);

  if (nullify_out_of_bounds) {
    gather_bitmask<true>(*source_table_view,
                         gather_map_begin,
                         masks.data().get(),
                         masks.size(),
                         num_destination_rows,
                         valid_counts.data().get(),
                         stream);
  } else {
    gather_bitmask<false>(*source_table_view,
                          gather_map_begin,
                          masks.data().get(),
                          masks.size(),
                          num_destination_rows,
                          valid_counts.data().get(),
                          stream);
  }

  thrust::host_vector<cudf::size_type> h_valid_counts(valid_counts);

  for (auto i=0; i<destination_table->num_columns(); ++i) {
    if (destination_table->get_column(i).nullable()) {
      destination_table->get_column(i).set_null_count(destination_table->num_rows()
                                                      - h_valid_counts[i]);
    }
  }

  return destination_table;
}


} // namespace detail
} // namespace experimental
} // namespace cudf
