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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/traits.hpp>

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
                       cudaStream_t stream)
{
  using MapValueType = typename thrust::iterator_traits<MapIterator>::value_type;

  // The gather_map is initialized with gather_rows value to identify pass-through entries
  // when calling the gather_bitmask() which applies a pass-through whenever it finds a
  // value outside the range of the target column.
  // We'll use the gather_rows value for this since it should always be outside the valid range.
  auto gather_map = rmm::device_vector<size_type>(gather_rows, gather_rows);

  // Convert scatter map to a gather map
  thrust::scatter(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<MapValueType>(0),
    thrust::make_counting_iterator<MapValueType>(std::distance(scatter_map_begin, scatter_map_end)),
    scatter_map_begin,
    gather_map.begin());

  return gather_map;
}

template <typename Element, typename MapIterator>
struct column_scatterer_impl {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    auto result      = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    using Type = device_storage_type_t<Element>;

    // NOTE use source.begin + scatter rows rather than source.end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    source.begin<Type>(),
                    source.begin<Type>() + cudf::distance(scatter_map_begin, scatter_map_end),
                    scatter_map_begin,
                    result_view.begin<Type>());

    return result;
  }
};

template <typename MapIterator>
struct column_scatterer_impl<string_view, MapIterator> {
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    using strings::detail::create_string_vector_from_column;
    auto const source_vector = create_string_vector_from_column(source, stream);
    auto const begin         = source_vector.begin();
    auto const end           = begin + std::distance(scatter_map_begin, scatter_map_end);
    return strings::detail::scatter(begin, end, scatter_map_begin, target, mr, stream);
  }
};

template <typename MapIterator>
struct column_scatterer {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    column_scatterer_impl<Element, MapIterator> scatterer{};
    return scatterer(source, scatter_map_begin, scatter_map_end, target, mr, stream);
  }
};

template <typename MapIterator>
struct column_scatterer_impl<dictionary32, MapIterator> {
  std::unique_ptr<column> operator()(column_view const& source_in,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target_in,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    if (target_in.size() == 0)  // empty begets empty
      return make_empty_column(data_type{type_id::DICTIONARY32});
    if (source_in.size() == 0)  // no input, just make a copy
      return std::make_unique<column>(target_in, stream, mr);

    // check the keys match
    dictionary_column_view const source(source_in);
    dictionary_column_view const target(target_in);
    CUDF_EXPECTS(source.keys().type() == target.keys().type(),
                 "scatter dictionary keys must be the same type");

    // first combine keys so both dictionaries have the same set
    auto target_matched    = dictionary::detail::add_keys(target, source.keys(), mr, stream);
    auto const target_view = dictionary_column_view(target_matched->view());
    auto source_matched    = dictionary::detail::set_keys(
      source, target_view.keys(), rmm::mr::get_current_device_resource(), stream);
    auto const source_view = dictionary_column_view(source_matched->view());

    // now build the new indices by doing a scatter on just the matched indices
    auto source_itr  = indexalator_factory::make_input_iterator(source_view.indices());
    auto new_indices = std::make_unique<column>(target_view.get_indices_annotated(), stream, mr);
    auto target_itr  = indexalator_factory::make_output_iterator(new_indices->mutable_view());
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
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
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Result of scattering values from source to target
 **/
template <typename MapIterator>
std::unique_ptr<table> scatter(
  table_view const& source,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_FUNC_RANGE();

  using MapType = typename thrust::iterator_traits<MapIterator>::value_type;

  if (check_bounds) {
    auto const begin = -target.num_rows();
    auto const end   = target.num_rows();
    auto bounds      = bounds_checker<MapType>{begin, end};
    CUDF_EXPECTS(
      std::distance(scatter_map_begin, scatter_map_end) ==
        thrust::count_if(
          rmm::exec_policy(stream)->on(stream), scatter_map_begin, scatter_map_end, bounds),
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
                   return type_dispatcher(source_col.type(),
                                          scatter_functor,
                                          source_col,
                                          updated_scatter_map_begin,
                                          updated_scatter_map_end,
                                          target_col,
                                          mr,
                                          stream);
                 });

  auto gather_map = scatter_to_gather(
    updated_scatter_map_begin, updated_scatter_map_end, target.num_rows(), stream);

  gather_bitmask(source, gather_map.begin(), result, gather_bitmask_op::PASSTHROUGH, mr, stream);

  return std::make_unique<table>(std::move(result));
}
}  // namespace detail
}  // namespace cudf
