/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/scatter.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

#include <cudf_test/column_utilities.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Convert a scatter map into a gather map.
 *
 * The caller is expected to use the output map on a subsequent gather_bitmask()
 * function using the PASSTHROUGH op since the resulting map may contain index
 * values outside the target's range.
 *
 * First, the gather-map is initialized with invalid entries.
 * The gather_rows is used since it should always be outside the target size.
 *
 * Then, the `output[scatter_map[i]] = i`.
 *
 * @tparam MapIterator Iterator type of the input scatter map.
 * @param scatter_map_begin Beginning of scatter map.
 * @param scatter_map_end End of the scatter map.
 * @param gather_rows Number of rows in the output map.
 * @param stream Stream used for CUDA kernel calls.
 * @return Output gather map.
 */
template <typename MapIterator>
auto scatter_to_gather(MapIterator scatter_map_begin,
                       MapIterator scatter_map_end,
                       size_type gather_rows,
                       rmm::cuda_stream_view stream)
{
  using MapValueType = typename thrust::iterator_traits<MapIterator>::value_type;

  // The gather_map is initialized with gather_rows value to identify pass-through entries
  // when calling the gather_bitmask() which applies a pass-through whenever it finds a
  // value outside the range of the target column.
  // We'll use the gather_rows value for this since it should always be outside the valid range.
  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), gather_map.begin(), gather_map.end(), gather_rows);

  // Convert scatter map to a gather map
  thrust::scatter(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<MapValueType>(0),
    thrust::make_counting_iterator<MapValueType>(std::distance(scatter_map_begin, scatter_map_end)),
    scatter_map_begin,
    gather_map.begin());

  return gather_map;
}

template <typename MapIterator>
rmm::device_uvector<size_type> scatter_to_gather_inv(MapIterator scatter_map_begin,
                                                     MapIterator scatter_map_end,
                                                     size_type gather_rows,
                                                     rmm::cuda_stream_view stream)
{
  using MapValueType = typename thrust::iterator_traits<MapIterator>::value_type;

  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::sequence(rmm::exec_policy(stream), gather_map.begin(), gather_map.end(), 0);

  // Convert scatter map to a gather map
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<MapValueType>(0),
    thrust::make_counting_iterator<MapValueType>(std::distance(scatter_map_begin, scatter_map_end)),
    [gather_rows, ptr = gather_map.begin(), scatter_map_begin = scatter_map_begin] __device__(
      MapValueType idx) {
      MapValueType row = *(scatter_map_begin + idx);
      ptr[row]         = gather_rows;
    });

  return gather_map;
}

template <typename Element, typename MapIterator>
struct column_scatterer_impl {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result      = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // NOTE use source.begin + scatter rows rather than source.end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy(stream),
                    source.begin<Element>(),
                    source.begin<Element>() + cudf::distance(scatter_map_begin, scatter_map_end),
                    scatter_map_begin,
                    result_view.begin<Element>());

    return result;
  }
};

template <typename MapIterator>
struct column_scatterer_impl<string_view, MapIterator> {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto d_column    = column_device_view::create(source, stream);
    auto const begin = d_column->begin<string_view>();
    auto const end   = begin + cudf::distance(scatter_map_begin, scatter_map_end);
    return strings::detail::scatter(begin, end, scatter_map_begin, target, stream, mr);
  }
};

template <typename MapIterator>
struct column_scatterer_impl<list_view, MapIterator> {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return cudf::lists::detail::scatter(
      source, scatter_map_begin, scatter_map_end, target, stream, mr);
  }
};

template <typename MapIterator>
struct column_scatterer {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    column_scatterer_impl<Element, MapIterator> scatterer{};
    return scatterer(source, scatter_map_begin, scatter_map_end, target, stream, mr);
  }
};

template <typename MapIterator>
struct column_scatterer_impl<dictionary32, MapIterator> {
  std::unique_ptr<column> operator()(column_view const& source_in,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target_in,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    if (target_in.is_empty())  // empty begets empty
      return make_empty_column(data_type{type_id::DICTIONARY32});
    if (source_in.is_empty())  // no input, just make a copy
      return std::make_unique<column>(target_in, stream, mr);

    // check the keys match
    dictionary_column_view const source(source_in);
    dictionary_column_view const target(target_in);
    CUDF_EXPECTS(source.keys().type() == target.keys().type(),
                 "scatter dictionary keys must be the same type");

    // first combine keys so both dictionaries have the same set
    auto target_matched    = dictionary::detail::add_keys(target, source.keys(), stream, mr);
    auto const target_view = dictionary_column_view(target_matched->view());
    auto source_matched    = dictionary::detail::set_keys(source, target_view.keys(), stream);
    auto const source_view = dictionary_column_view(source_matched->view());

    // now build the new indices by doing a scatter on just the matched indices
    auto source_itr  = indexalator_factory::make_input_iterator(source_view.indices());
    auto new_indices = std::make_unique<column>(target_view.get_indices_annotated(), stream, mr);
    auto target_itr  = indexalator_factory::make_output_iterator(new_indices->mutable_view());
    thrust::scatter(rmm::exec_policy(stream),
                    source_itr,
                    source_itr + std::distance(scatter_map_begin, scatter_map_end),
                    scatter_map_begin,
                    target_itr);

    // record some data before calling release()
    auto const indices_type = new_indices->type();
    auto const output_size  = new_indices->size();
    auto const null_count   = new_indices->null_count();
    auto contents           = new_indices->release();
    auto indices_column     = std::make_unique<column>(indices_type,
                                                   static_cast<size_type>(output_size),
                                                   std::move(*(contents.data.release())),
                                                   rmm::device_buffer{0, stream, mr},
                                                   0);

    // take the keys from the matched column allocated using mr
    std::unique_ptr<column> keys_column(std::move(target_matched->release().children.back()));

    // create column with keys_column and indices_column
    return make_dictionary_column(std::move(keys_column),
                                  std::move(indices_column),
                                  std::move(*(contents.null_mask.release())),
                                  null_count);
  }
};

template <typename MapItRoot>
struct column_scatterer_impl<struct_view, MapItRoot> {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapItRoot scatter_map_begin,
                                     MapItRoot scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(source.num_children() == target.num_children(),
                 "Scatter source and target are not of the same type.");

    auto const scatter_map_size = std::distance(scatter_map_begin, scatter_map_end);
    if (scatter_map_size == 0) { return empty_like(source); }

    structs_column_view structs_src(source);
    structs_column_view structs_target(target);
    std::vector<std::unique_ptr<column>> output_struct_members(structs_src.num_children());

    std::transform(structs_src.child_begin(),
                   structs_src.child_end(),
                   structs_target.child_begin(),
                   output_struct_members.begin(),
                   [&scatter_map_begin, &scatter_map_end, stream, mr](auto const& source_col,
                                                                      auto const& target_col) {
                     return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                   column_scatterer<MapItRoot>{},
                                                                   source_col,
                                                                   scatter_map_begin,
                                                                   scatter_map_end,
                                                                   target_col,
                                                                   stream,
                                                                   mr);
                   });

    auto const nullable = std::any_of(structs_src.child_begin(),
                                      structs_src.child_end(),
                                      [](auto const& col) { return col.nullable(); }) or
                          std::any_of(structs_target.child_begin(),
                                      structs_target.child_end(),
                                      [](auto const& col) { return col.nullable(); });
    if (nullable) {
      auto const gather_map =
        scatter_to_gather(scatter_map_begin, scatter_map_end, source.size(), stream);

      int n = (int)std::distance(gather_map.begin(), gather_map.end());
      //      thrust::host_vector<int> h(gather_map.begin(), gather_map.end());
      printf("\n\n");
      //      for (int i = 0; i < n; ++i) { printf("h: %d\n", gather_map.element(i, stream)); }

      printf("line %d \n\n", __LINE__);
      cudf::test::print(*structs_src.child_begin());

      printf("line %d \n\n", __LINE__);
      cudf::test::print(*structs_target.child_begin());

      printf("line %d \n\n", __LINE__);
      cudf::test::print((*output_struct_members.begin())->view());

      printf("map siE:%d\n", (int)std::distance(gather_map.begin(), gather_map.end()));
      printf("\n\n");
      printf("num row: %d\n", source.size());
      printf("num c row: %d\n", structs_src.child_begin()->size());

      printf("source null count: %d\n", (*structs_src.child_begin()).null_count());
      printf("target null count: %d\n", (*structs_target.child_begin()).null_count());
      printf("result null count: %d\n", (*output_struct_members.begin())->view().null_count());

      gather_bitmask(cudf::table_view{std::vector<cudf::column_view>{structs_src.child_begin(),
                                                                     structs_src.child_end()}},
                     gather_map.begin(),
                     output_struct_members,
                     gather_bitmask_op::PASSTHROUGH,
                     stream,
                     mr);

      printf("result null count again: %d\n",
             (*output_struct_members.begin())->view().null_count());

      printf("line %d \n\n", __LINE__);
      cudf::test::print((*output_struct_members.begin())->view());
    }

    std::vector<std::unique_ptr<column>> result;
    result.emplace_back(cudf::make_structs_column(
      source.size(),
      std::move(output_struct_members),
      0,
      rmm::device_buffer{0, stream, mr},  // Null mask will be fixed up in cudf::scatter().
      stream,
      mr));

    // Only gather bitmask from the target at the positions that have not been scatter onto
    auto const gather_map =
      scatter_to_gather_inv(scatter_map_begin, scatter_map_end, source.size(), stream);
    gather_bitmask(table_view{std::vector<cudf::column_view>{target}},
                   gather_map.begin(),
                   result,
                   gather_bitmask_op::PASSTHROUGH,
                   stream,
                   mr);

    return std::move(result.front());

    // std::vector<std::unique_ptr<column>> output_struct_members(structs_src.num_children());
    //    for (auto& col : output_struct_members) { col->set_null_count(0); }
    //
    //    return cudf::make_structs_column(
    //      source.size(),
    //      std::move(output_struct_members),
    //      target.null_count(),
    //      cudf::detail::copy_bitmask(
    //        target, stream, mr),  // Null mask will be fixed up in cudf::scatter().
    //      stream,
    //      mr);
  }
};

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_begin[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined. This range might have negative values, which will be modified by adding target.size()
 *
 * @throws cudf::logic_error if scatter map index is out of bounds
 * @throws cudf::logic_error if scatter_map.size() > source.num_rows()
 *
 * @param[in] source The input columns containing values to be scattered into the
 * target columns
 * @param[in] scatter_map_begin Beginning of iterator range of integer indices that has been
 *provided.
 * @param[in] scatter_map_end End of iterator range of integer indices that has been provided.
 * source columns to rows in the target columns
 * @param[in] target The set of columns into which values from the source_table
 * are to be scattered
 * @param[in] check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Result of scattering values from source to target
 */
template <typename MapIterator>
std::unique_ptr<table> scatter(
  table_view const& source,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_FUNC_RANGE();

  using MapType = typename thrust::iterator_traits<MapIterator>::value_type;

  if (check_bounds) {
    auto const begin = -target.num_rows();
    auto const end   = target.num_rows();
    auto bounds      = bounds_checker<MapType>{begin, end};
    CUDF_EXPECTS(
      std::distance(scatter_map_begin, scatter_map_end) ==
        thrust::count_if(rmm::exec_policy(stream), scatter_map_begin, scatter_map_end, bounds),
      "Scatter map index out of bounds");
  }

  CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) <= source.num_rows(),
               "scatter map size should be <= to number of rows in source");

  // Transform negative indices to index + target size
  auto updated_scatter_map_begin =
    thrust::make_transform_iterator(scatter_map_begin, index_converter<MapType>{target.num_rows()});

  auto updated_scatter_map_end =
    thrust::make_transform_iterator(scatter_map_end, index_converter<MapType>{target.num_rows()});

  auto result = std::vector<std::unique_ptr<column>>(target.num_columns());

  auto scatter_functor = column_scatterer<decltype(updated_scatter_map_begin)>{};

  std::transform(source.begin(),
                 source.end(),
                 target.begin(),
                 result.begin(),
                 [=](auto const& source_col, auto const& target_col) {
                   return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                 scatter_functor,
                                                                 source_col,
                                                                 updated_scatter_map_begin,
                                                                 updated_scatter_map_end,
                                                                 target_col,
                                                                 stream,
                                                                 mr);
                 });

  printf("line %d \n\n", __LINE__);

  auto const nullable =
    std::any_of(source.begin(), source.end(), [](auto const& col) { return col.nullable(); }) or
    std::any_of(target.begin(), target.end(), [](auto const& col) { return col.nullable(); });
  if (nullable) {
    printf("nullable\n");
    auto gather_map = scatter_to_gather(
      updated_scatter_map_begin, updated_scatter_map_end, target.num_rows(), stream);

    int n = (int)std::distance(gather_map.begin(), gather_map.end());
    //      thrust::host_vector<int> h(gather_map.begin(), gather_map.end());
    printf("\n\n");
    //    for (int i = 0; i < n; ++i) { printf("gather map: %d\n", gather_map.element(i, stream)); }
    printf("source null count: %d\n", (*source.begin()).null_count());
    printf("target null count: %d\n", (*target.begin()).null_count());
    printf("result null count: %d\n", (*result.begin())->null_count());

    gather_bitmask(source, gather_map.begin(), result, gather_bitmask_op::PASSTHROUGH, stream, mr);

    printf("result null count again: %d\n", (*result.begin())->null_count());
  } else
    printf("no t nullable\n");
  return std::make_unique<table>(std::move(result));
}
}  // namespace detail
}  // namespace cudf
