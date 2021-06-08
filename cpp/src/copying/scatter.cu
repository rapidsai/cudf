/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/lists/list_view.cuh>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace detail {
namespace {

template <bool mark_true, typename MapIterator>
__global__ void marking_bitmask_kernel(mutable_column_device_view destination,
                                       MapIterator scatter_map,
                                       size_type num_scatter_rows)
{
  size_type row = threadIdx.x + blockIdx.x * blockDim.x;

  while (row < num_scatter_rows) {
    size_type const output_row = scatter_map[row];

    if (mark_true) {
      destination.set_valid(output_row);
    } else {
      destination.set_null(output_row);
    }

    row += blockDim.x * gridDim.x;
  }
}

template <typename MapIterator>
void scatter_scalar_bitmask(std::vector<std::reference_wrapper<const scalar>> const& source,
                            MapIterator scatter_map,
                            size_type num_scatter_rows,
                            std::vector<std::unique_ptr<column>>& target,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  constexpr size_type block_size = 256;
  size_type const grid_size      = grid_1d(num_scatter_rows, block_size).num_blocks;

  for (size_t i = 0; i < target.size(); ++i) {
    auto const source_is_valid = source[i].get().is_valid(stream);
    if (target[i]->nullable() or not source_is_valid) {
      if (not target[i]->nullable()) {
        // Target must have a null mask if the source is not valid
        auto mask = detail::create_null_mask(target[i]->size(), mask_state::ALL_VALID, stream, mr);
        target[i]->set_null_mask(std::move(mask), 0);
      }

      auto target_view = mutable_column_device_view::create(target[i]->mutable_view(), stream);

      auto bitmask_kernel = source_is_valid ? marking_bitmask_kernel<true, decltype(scatter_map)>
                                            : marking_bitmask_kernel<false, decltype(scatter_map)>;
      bitmask_kernel<<<grid_size, block_size, 0, stream.value()>>>(
        *target_view, scatter_map, num_scatter_rows);
    }
  }
}

template <typename Element, typename MapIterator>
struct column_scalar_scatterer_impl {
  std::unique_ptr<column> operator()(std::reference_wrapper<const scalar> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(source.get().type() == target.type(), "scalar and column types must match");

    auto result      = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // Use permutation iterator with constant index to dereference scalar data
    auto scalar_impl = static_cast<const scalar_type_t<Element>*>(&source.get());
    auto scalar_iter =
      thrust::make_permutation_iterator(scalar_impl->data(), thrust::make_constant_iterator(0));

    thrust::scatter(rmm::exec_policy(stream),
                    scalar_iter,
                    scalar_iter + scatter_rows,
                    scatter_iter,
                    result_view.begin<Element>());

    return result;
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<string_view, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<const scalar> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(source.get().type() == target.type(), "scalar and column types must match");

    auto const scalar_impl = static_cast<const string_scalar*>(&source.get());
    auto const source_view = string_view(scalar_impl->data(), scalar_impl->size());
    auto const begin       = thrust::make_constant_iterator(source_view);
    auto const end         = begin + scatter_rows;
    return strings::detail::scatter(begin, end, scatter_iter, target, stream, mr);
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<list_view, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<const scalar> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return lists::detail::scatter(
      source, scatter_iter, scatter_iter + scatter_rows, target, stream, mr);
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<struct_view, MapIterator> {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
  {
    CUDF_FAIL("scatter scalar to struct_view not implemented");
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<dictionary32, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<const scalar> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto dict_target =
      dictionary::detail::add_keys(dictionary_column_view(target),
                                   make_column_from_scalar(source.get(), 1, stream)->view(),
                                   stream,
                                   mr);
    auto dict_view    = dictionary_column_view(dict_target->view());
    auto scalar_index = dictionary::detail::get_index(dict_view, source.get(), stream);
    auto scalar_iter  = thrust::make_permutation_iterator(
      indexalator_factory::make_input_iterator(*scalar_index), thrust::make_constant_iterator(0));
    auto new_indices = std::make_unique<column>(dict_view.get_indices_annotated(), stream, mr);
    auto target_iter = indexalator_factory::make_output_iterator(new_indices->mutable_view());

    thrust::scatter(
      rmm::exec_policy(stream), scalar_iter, scalar_iter + scatter_rows, scatter_iter, target_iter);

    // build the dictionary indices column from the result
    auto const indices_type = new_indices->type();
    auto const output_size  = new_indices->size();
    auto const null_count   = new_indices->null_count();
    auto contents           = new_indices->release();
    auto indices_column     = std::make_unique<column>(indices_type,
                                                   static_cast<size_type>(output_size),
                                                   std::move(*(contents.data.release())),
                                                   rmm::device_buffer{},
                                                   0);
    // use the keys from the matched column
    std::unique_ptr<column> keys_column(std::move(dict_target->release().children.back()));
    // create the output column
    return make_dictionary_column(std::move(keys_column),
                                  std::move(indices_column),
                                  std::move(*(contents.null_mask.release())),
                                  null_count);
  }
};

template <typename MapIterator>
struct column_scalar_scatterer {
  template <typename Element>
  std::unique_ptr<column> operator()(std::reference_wrapper<const scalar> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    column_scalar_scatterer_impl<Element, MapIterator> scatterer{};
    return scatterer(source, scatter_iter, scatter_rows, target, stream, mr);
  }
};

}  // namespace

std::unique_ptr<table> scatter(table_view const& source,
                               column_view const& scatter_map,
                               table_view const& target,
                               bool check_bounds,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(source.num_columns() == target.num_columns(),
               "Number of columns in source and target not equal");
  CUDF_EXPECTS(scatter_map.size() <= source.num_rows(),
               "Size of scatter map must be equal to or less than source rows");
  CUDF_EXPECTS(std::equal(source.begin(),
                          source.end(),
                          target.begin(),
                          [](auto const& col1, auto const& col2) {
                            return col1.type().id() == col2.type().id();
                          }),
               "Column types do not match between source and target");
  CUDF_EXPECTS(scatter_map.has_nulls() == false, "Scatter map contains nulls");

  if (scatter_map.is_empty()) { return std::make_unique<table>(target, stream, mr); }

  // create index type normalizing iterator for the scatter_map
  auto map_begin = indexalator_factory::make_input_iterator(scatter_map);
  auto map_end   = map_begin + scatter_map.size();
  return detail::scatter(source, map_begin, map_end, target, check_bounds, stream, mr);
}

std::unique_ptr<table> scatter(std::vector<std::reference_wrapper<const scalar>> const& source,
                               column_view const& indices,
                               table_view const& target,
                               bool check_bounds,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(source.size() == static_cast<size_t>(target.num_columns()),
               "Number of columns in source and target not equal");
  CUDF_EXPECTS(indices.has_nulls() == false, "indices contains nulls");

  if (indices.is_empty()) { return std::make_unique<table>(target, stream, mr); }

  // Create normalizing iterator for indices column
  auto map_begin = indexalator_factory::make_input_iterator(indices);
  auto map_end   = map_begin + indices.size();

  // Optionally check map index values are within the number of target rows.
  auto const n_rows = target.num_rows();
  if (check_bounds) {
    CUDF_EXPECTS(
      indices.size() == thrust::count_if(rmm::exec_policy(stream),
                                         map_begin,
                                         map_end,
                                         [n_rows] __device__(size_type index) {
                                           return ((index >= -n_rows) && (index < n_rows));
                                         }),
      "Scatter map index out of bounds");
  }

  // Transform negative indices to index + target size
  auto scatter_rows = indices.size();
  auto scatter_iter = thrust::make_transform_iterator(
    map_begin, [n_rows] __device__(size_type in) { return ((in % n_rows) + n_rows) % n_rows; });

  // Dispatch over data type per column
  auto result          = std::vector<std::unique_ptr<column>>(target.num_columns());
  auto scatter_functor = column_scalar_scatterer<decltype(scatter_iter)>{};
  std::transform(source.begin(),
                 source.end(),
                 target.begin(),
                 result.begin(),
                 [=](auto const& source_scalar, auto const& target_col) {
                   return type_dispatcher<dispatch_storage_type>(target_col.type(),
                                                                 scatter_functor,
                                                                 source_scalar,
                                                                 scatter_iter,
                                                                 scatter_rows,
                                                                 target_col,
                                                                 stream,
                                                                 mr);
                 });

  scatter_scalar_bitmask(source, scatter_iter, scatter_rows, result, stream, mr);

  return std::make_unique<table>(std::move(result));
}

std::unique_ptr<column> boolean_mask_scatter(column_view const& input,
                                             column_view const& target,
                                             column_view const& boolean_mask,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, target.size(), mask_state::UNALLOCATED, stream);
  auto mutable_indices = indices->mutable_view();

  thrust::sequence(rmm::exec_policy(stream),
                   mutable_indices.begin<size_type>(),
                   mutable_indices.end<size_type>(),
                   0);

  // The scatter map is actually a table with only one column, which is scatter map.
  auto scatter_map =
    detail::apply_boolean_mask(table_view{{indices->view()}}, boolean_mask, stream);
  auto output_table = detail::scatter(table_view{{input}},
                                      scatter_map->get_column(0).view(),
                                      table_view{{target}},
                                      false,
                                      stream,
                                      mr);

  // There is only one column in output_table
  return std::make_unique<column>(std::move(output_table->get_column(0)));
}

std::unique_ptr<column> boolean_mask_scatter(scalar const& input,
                                             column_view const& target,
                                             column_view const& boolean_mask,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::copy_if_else(input, target, boolean_mask, stream, mr);
}

std::unique_ptr<table> boolean_mask_scatter(table_view const& input,
                                            table_view const& target,
                                            column_view const& boolean_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.num_columns() == target.num_columns(),
               "Mismatch in number of input columns and target columns");
  CUDF_EXPECTS(boolean_mask.size() == target.num_rows(),
               "Boolean mask size and number of target rows mismatch");
  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be of Boolean type");
  // Count valid pair of input and columns as per type at each column index i
  CUDF_EXPECTS(
    std::all_of(thrust::counting_iterator<size_type>(0),
                thrust::counting_iterator<size_type>(target.num_columns()),
                [&input, &target](auto index) {
                  return ((input.column(index).type().id()) == (target.column(index).type().id()));
                }),
    "Type mismatch in input column and target column");

  if (target.num_rows() != 0) {
    std::vector<std::unique_ptr<column>> out_columns(target.num_columns());
    std::transform(
      input.begin(),
      input.end(),
      target.begin(),
      out_columns.begin(),
      [&boolean_mask, mr, stream](auto const& input_column, auto const& target_column) {
        return boolean_mask_scatter(input_column, target_column, boolean_mask, stream, mr);
      });

    return std::make_unique<table>(std::move(out_columns));
  } else {
    return empty_like(target);
  }
}

std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<const scalar>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(static_cast<size_type>(input.size()) == target.num_columns(),
               "Mismatch in number of scalars and target columns");
  CUDF_EXPECTS(boolean_mask.size() == target.num_rows(),
               "Boolean mask size and number of target rows mismatch");
  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be of Boolean type");

  // Count valid pair of input and columns as per type at each column/scalar index i
  CUDF_EXPECTS(
    std::all_of(thrust::counting_iterator<size_type>(0),
                thrust::counting_iterator<size_type>(target.num_columns()),
                [&input, &target](auto index) {
                  return (input[index].get().type().id() == target.column(index).type().id());
                }),
    "Type mismatch in input scalar and target column");

  if (target.num_rows() != 0) {
    std::vector<std::unique_ptr<column>> out_columns(target.num_columns());
    std::transform(input.begin(),
                   input.end(),
                   target.begin(),
                   out_columns.begin(),
                   [&boolean_mask, mr, stream](auto const& scalar, auto const& target_column) {
                     return boolean_mask_scatter(
                       scalar.get(), target_column, boolean_mask, stream, mr);
                   });

    return std::make_unique<table>(std::move(out_columns));
  } else {
    return empty_like(target);
  }
}

}  // namespace detail

std::unique_ptr<table> scatter(table_view const& source,
                               column_view const& scatter_map,
                               table_view const& target,
                               bool check_bounds,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scatter(source, scatter_map, target, check_bounds, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> scatter(std::vector<std::reference_wrapper<const scalar>> const& source,
                               column_view const& indices,
                               table_view const& target,
                               bool check_bounds,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scatter(source, indices, target, check_bounds, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> boolean_mask_scatter(table_view const& input,
                                            table_view const& target,
                                            column_view const& boolean_mask,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::boolean_mask_scatter(input, target, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<const scalar>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::boolean_mask_scatter(input, target, boolean_mask, rmm::cuda_stream_default, mr);
}

}  // namespace cudf

//
//
//
//
//
//
//

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

namespace cudf {
namespace detail {

/**
 * @brief Convert a scatter map into a gather map.
 *
 * The caller is expected to use the output map on a subsequent gather_bitmask()
 * function using the PASSTHROUGH op since the resulting map may contain index
 * values outside the target's range.
 *
 * First, the gather-map is initialized with an invalid index.
 * The value `numeric_limits::lowest()` is used since it should always be outside the target size.
 * Then, `output[scatter_map[i]] = i` for each `i`.
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

  // The gather_map is initialized with `numeric_limits::lowest()` value to identify pass-through
  // entries when calling the gather_bitmask() which applies a pass-through whenever it finds a
  // value outside the range of the target column.
  // We'll use the `numeric_limits::lowest()` value for this since it should always be outside the
  // valid range.
  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             gather_map.begin(),
                             gather_map.end(),
                             std::numeric_limits<size_type>::lowest());

  // Convert scatter map to a gather map
  thrust::scatter(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<MapValueType>(0),
    thrust::make_counting_iterator<MapValueType>(std::distance(scatter_map_begin, scatter_map_end)),
    scatter_map_begin,
    gather_map.begin());

  return gather_map;
}

/**
 * @brief Create a complement map of `scatter_to_gather` map
 *
 * The purpose of this map is to create an identity-mapping for the rows that are not
 * touched by the `scatter_map`.
 *
 * The output result of this mapping is firstly initialized as an identity-mapping
 * (i.e., `output[i] = i`). Then, for each value `idx` from `scatter_map`, the value `output[idx]`
 * is set to `numeric_limits::lowest()`, which is an invalid, out-of-bound index to identify the
 * pass-through entries when calling the `gather_bitmask()` function.
 *
 */
template <typename MapIterator>
auto scatter_to_gather_complement(MapIterator scatter_map_begin,
                                  MapIterator scatter_map_end,
                                  size_type gather_rows,
                                  rmm::cuda_stream_view stream)
{
  auto gather_map = rmm::device_uvector<size_type>(gather_rows, stream);
  thrust::sequence(rmm::exec_policy(stream), gather_map.begin(), gather_map.end(), 0);

  auto const out_of_bounds_begin =
    thrust::make_constant_iterator(std::numeric_limits<size_type>::lowest());
  auto const out_of_bounds_end =
    out_of_bounds_begin + thrust::distance(scatter_map_begin, scatter_map_end);
  thrust::scatter(rmm::exec_policy(stream),
                  out_of_bounds_begin,
                  out_of_bounds_end,
                  scatter_map_begin,
                  gather_map.begin());
  return gather_map;
}

template <typename Element, typename Enable = void>
struct column_scatterer_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported type for scatter.");
  }
};

template <typename Element>
struct column_scatterer_impl<Element, std::enable_if_t<cudf::is_fixed_width<Element>()>> {
  template <typename MapIterator>
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

template <>
struct column_scatterer_impl<string_view> {
  template <typename MapIterator>
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

template <>
struct column_scatterer_impl<list_view> {
  template <typename MapIterator>
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

template <>
struct column_scatterer_impl<dictionary32> {
  template <typename MapIterator>
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

struct column_scatterer {
  template <typename Element, typename MapIterator>
  std::unique_ptr<column> operator()(column_view const& source,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    column_scatterer_impl<Element> scatterer{};
    return scatterer(source, scatter_map_begin, scatter_map_end, target, stream, mr);
  }
};

template <>
struct column_scatterer_impl<struct_view> {
  template <typename MapItRoot>
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
    if (scatter_map_size == 0) { return std::make_unique<column>(target, stream, mr); }

    structs_column_view const structs_src(source);
    structs_column_view const structs_target(target);
    std::vector<std::unique_ptr<column>> output_struct_members(structs_src.num_children());

    std::transform(structs_src.child_begin(),
                   structs_src.child_end(),
                   structs_target.child_begin(),
                   output_struct_members.begin(),
                   [&scatter_map_begin, &scatter_map_end, stream, mr](auto const& source_col,
                                                                      auto const& target_col) {
                     return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                   column_scatterer{},
                                                                   source_col,
                                                                   scatter_map_begin,
                                                                   scatter_map_end,
                                                                   target_col,
                                                                   stream,
                                                                   mr);
                   });

    // We still need to call `gather_bitmask` even when the source's children are not nullable,
    // as if the target's children have null_masks, those null_masks need to be updated after
    // being scattered onto.
    auto const child_nullable = std::any_of(structs_src.child_begin(),
                                            structs_src.child_end(),
                                            [](auto const& col) { return col.nullable(); }) or
                                std::any_of(structs_target.child_begin(),
                                            structs_target.child_end(),
                                            [](auto const& col) { return col.nullable(); });
    if (child_nullable) {
      auto const gather_map =
        scatter_to_gather(scatter_map_begin, scatter_map_end, source.size(), stream);
      gather_bitmask(cudf::table_view{std::vector<cudf::column_view>{structs_src.child_begin(),
                                                                     structs_src.child_end()}},
                     gather_map.begin(),
                     output_struct_members,
                     gather_bitmask_op::PASSTHROUGH,
                     stream,
                     mr);
    }

    // Need to put the result column in a vector to call `gather_bitmask`.
    std::vector<std::unique_ptr<column>> result;
    result.emplace_back(cudf::make_structs_column(target.size(),
                                                  std::move(output_struct_members),
                                                  0,
                                                  rmm::device_buffer{0, stream, mr},
                                                  stream,
                                                  mr));

    // Only gather bitmask from the target column for the rows that have not been scattered onto
    // The bitmask from the source column will be gathered at the top level `scatter()` call.
    if (target.nullable()) {
      auto const gather_map =
        scatter_to_gather_complement(scatter_map_begin, scatter_map_end, target.size(), stream);
      gather_bitmask(table_view{std::vector<cudf::column_view>{target}},
                     gather_map.begin(),
                     result,
                     gather_bitmask_op::PASSTHROUGH,
                     stream,
                     mr);
    }

    return std::move(result.front());
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
std::unique_ptr<table> scatter(table_view const& source,
                               MapIterator scatter_map_begin,
                               MapIterator scatter_map_end,
                               table_view const& target,
                               bool check_bounds,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  using MapType = typename thrust::iterator_traits<MapIterator>::value_type;

  if (check_bounds) {
    auto const begin = -target.num_rows();
    auto const end   = target.num_rows();
    CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) ==
                   thrust::count_if(rmm::exec_policy(stream),
                                    scatter_map_begin,
                                    scatter_map_end,
                                    [begin, end] __device__(auto const idx) {
                                      return ((idx >= begin) && (idx < end));
                                    }),
                 "Scatter map index out of bounds");
  }

  CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) <= source.num_rows(),
               "scatter map size should be <= to number of rows in source");

  // Transform negative indices to index + target size.
  auto updated_scatter_map_begin =
    thrust::make_transform_iterator(scatter_map_begin, index_converter<MapType>{target.num_rows()});
  auto updated_scatter_map_end =
    thrust::make_transform_iterator(scatter_map_end, index_converter<MapType>{target.num_rows()});
  auto result = std::vector<std::unique_ptr<column>>(target.num_columns());

  std::transform(source.begin(),
                 source.end(),
                 target.begin(),
                 result.begin(),
                 [=](auto const& source_col, auto const& target_col) {
                   return type_dispatcher<dispatch_storage_type>(source_col.type(),
                                                                 column_scatterer{},
                                                                 source_col,
                                                                 updated_scatter_map_begin,
                                                                 updated_scatter_map_end,
                                                                 target_col,
                                                                 stream,
                                                                 mr);
                 });

  // We still need to call `gather_bitmask` even when the source columns are not nullable,
  // as if the target has null_mask, that null_mask needs to be updated after scattering.
  auto const nullable =
    std::any_of(source.begin(), source.end(), [](auto const& col) { return col.nullable(); }) or
    std::any_of(target.begin(), target.end(), [](auto const& col) { return col.nullable(); });
  if (nullable) {
    auto const gather_map = scatter_to_gather(
      updated_scatter_map_begin, updated_scatter_map_end, target.num_rows(), stream);
    gather_bitmask(source, gather_map.begin(), result, gather_bitmask_op::PASSTHROUGH, stream, mr);

    // For struct columns, we need to superimpose the null_mask of the parent over the null_mask of
    // the children.
    std::for_each(result.begin(), result.end(), [=](auto& col) {
      auto const col_view = col->view();
      if (col_view.type().id() == type_id::STRUCT and col_view.nullable()) {
        auto const num_rows   = col_view.size();
        auto const null_count = col_view.null_count();
        auto contents         = col->release();

        // Children null_mask will be superimposed during structs column construction.
        col = cudf::make_structs_column(num_rows,
                                        std::move(contents.children),
                                        null_count,
                                        std::move(*contents.null_mask),
                                        stream,
                                        mr);
      }
    });
  }
  return std::make_unique<table>(std::move(result));
}

#define INSTANTIATE(MapIterator)                                              \
  template std::unique_ptr<table> scatter<MapIterator>(table_view const&,     \
                                                       MapIterator,           \
                                                       MapIterator,           \
                                                       table_view const&,     \
                                                       bool,                  \
                                                       rmm::cuda_stream_view, \
                                                       rmm::mr::device_memory_resource*);

INSTANTIATE(int8_t*)
INSTANTIATE(int16_t*)
INSTANTIATE(int32_t*)
INSTANTIATE(int64_t*)

INSTANTIATE(uint8_t*)
INSTANTIATE(uint16_t*)
INSTANTIATE(uint32_t*)
INSTANTIATE(uint64_t*)

}  // namespace detail
}  // namespace cudf
