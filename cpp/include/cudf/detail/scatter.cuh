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

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <memory>

namespace cudf {
namespace experimental {
namespace detail {

template <typename T, typename MapIterator>
rmm::device_vector<T> scatter_to_gather(MapIterator scatter_map_begin,
    MapIterator scatter_map_end, size_type gather_rows,
    cudaStream_t stream)
{
  static_assert(std::is_signed<T>::value,
    "Need different invalid index if unsigned index types are added");
  auto const invalid_index = static_cast<T>(-1);

  // Convert scatter map to a gather map
  auto gather_map = rmm::device_vector<T>(gather_rows, invalid_index);
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<T>(0),
    thrust::make_counting_iterator<T>(std::distance(scatter_map_begin, scatter_map_end)),
    scatter_map_begin, gather_map.begin());

  return gather_map;
}

template <typename Element, typename MapIterator>
struct column_scatterer_impl
{
//  template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto result = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // NOTE use source.begin + scatter rows rather than source.end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy(stream)->on(stream), source.begin<Element>(),
      source.begin<Element>() + std::distance(scatter_map_begin, scatter_map_end), 
      scatter_map_begin,
      result_view.begin<Element>());

    return result;
  }
};

//template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
template <typename MapIterator>
struct column_scatterer_impl<string_view, MapIterator>
{
  std::unique_ptr<column> operator()(column_view const& source,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    using strings::detail::create_string_vector_from_column;
    auto const source_vector = create_string_vector_from_column(source, stream);
    auto const begin = source_vector.begin();
    auto const end = begin + std::distance(scatter_map_begin, scatter_map_end);
    return strings::detail::scatter(begin, end, scatter_map_begin, target, mr, stream);
  }
};

template <typename MapIterator>
struct column_scatterer_impl<dictionary32, MapIterator>
{
  std::unique_ptr<column> operator()(column_view const& source_in,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target_in,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    if( target_in.size() == 0 )
        return make_empty_column(data_type{DICTIONARY32});
    auto output_count = std::distance(scatter_map_begin, scatter_map_end);
    if( source_in.size() == 0 || output_count == 0 )
        return std::make_unique<column>( source_in, stream, mr );
    dictionary_column_view source(source_in);
    dictionary_column_view target(target_in);
    CUDF_EXPECTS( source.keys().type()==target.keys().type(), "scatter dictionary keys must be the same type");

    // first combine keys so both dictionaries have the same set
    auto target_matched = dictionary::detail::add_keys(target,source.keys(),mr,stream);
    auto target_view = dictionary_column_view(target_matched->view());
    auto source_matched = dictionary::detail::set_keys(source,target_view.keys(),mr,stream);
    auto source_view = dictionary_column_view(source_matched->view());

    // now we can the build new indices by doing a scatter on them
    column_view source_indices( data_type{INT32}, source_view.size(),
                                source_view.indices().data<int32_t>(),
                                source_view.null_mask(), source_view.null_count(),
                                source_view.offset() );
    column_view target_indices( data_type{INT32}, target_view.size(),
                                target_view.indices().data<int32_t>(),
                                target_view.null_mask(), target_view.null_count(),
                                target_view.offset() );
    column_scatterer_impl<int32_t,MapIterator> index_scatterer;
    auto new_indices = index_scatterer( source_indices, scatter_map_begin, scatter_map_end, target_indices, mr, stream);
    auto null_count = new_indices->null_count();
    auto contents = new_indices->release();
    auto indices_column = std::make_unique<column>( data_type{INT32},
          static_cast<size_type>(output_count), std::move(*(contents.data.release())),
          rmm::device_buffer{}, 0 ); // set null count to 0

    auto keys_column = std::make_unique<column>( target_view.keys(), stream, mr );

    // create column with keys_column and indices_column
    return make_dictionary_column( std::move(keys_column), std::move(indices_column),
                                   std::move(*(contents.null_mask.release())), null_count );
  }
};

template <typename MapIterator>
struct column_scatterer
{
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& source,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
      column_scatterer_impl<Element, MapIterator> scatterer{};
      return scatterer(source, scatter_map_begin, scatter_map_end, target, mr, stream);
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
 * @param[in] scatter_map_begin Beginning of iterator range of integer indices that has been provided.
 * @param[in] scatter_map_end End of iterator range of integer indices that has been provided.
 * source columns to rows in the target columns
 * @param[in] target The set of columns into which values from the source_table
 * are to be scattered
 * @param[in] check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The stream to use for CUDA operations
 *
 * @return Result of scattering values from source to target
 **/
template <typename T, typename MapIterator>
std::unique_ptr<table>
scatter(table_view const& source, MapIterator scatter_map_begin,
    MapIterator scatter_map_end, table_view const& target,
    bool check_bounds = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0) {

    if (check_bounds) {
      auto const begin = -target.num_rows();
      auto const end = target.num_rows();
      auto bounds = bounds_checker<T>{begin, end};
      CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) == thrust::count_if(
        rmm::exec_policy(stream)->on(stream),
        scatter_map_begin, scatter_map_end, bounds),
        "Scatter map index out of bounds");
    }

    CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) <= source.num_rows(),
            "scatter map size should be <= to number of rows in source");

    // Transform negative indices to index + target size
    auto updated_scatter_map_begin = thrust::make_transform_iterator(
                           scatter_map_begin,
                           index_converter<T>{target.num_rows()});

    auto updated_scatter_map_end = thrust::make_transform_iterator(
                           scatter_map_end,
                           index_converter<T>{target.num_rows()});

    auto result = std::vector<std::unique_ptr<column>>(target.num_columns());
    auto scatter_functor = column_scatterer<decltype(updated_scatter_map_begin)>{};
    std::transform(source.begin(), source.end(), target.begin(), result.begin(),
      [=](auto const& source_col, auto const& target_col) {
        return type_dispatcher(source_col.type(), scatter_functor,
          source_col, updated_scatter_map_begin, 
          updated_scatter_map_end, target_col, mr, stream);
      });

    auto gather_map = scatter_to_gather<T>(updated_scatter_map_begin, updated_scatter_map_end,
      target.num_rows(), stream);
    gather_bitmask(source, gather_map.begin(), result,
      gather_bitmask_op::PASSTHROUGH, mr, stream);

    return std::make_unique<table>(std::move(result));
}
} //namespace detail
} //namespace experimental
} //namespace detail
