/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <stdexcept>

namespace cudf {
namespace detail {
namespace {

template <bool mark_true, typename MapIterator>
CUDF_KERNEL void marking_bitmask_kernel(mutable_column_device_view destination,
                                        MapIterator scatter_map,
                                        size_type num_scatter_rows)
{
  auto row          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  while (row < num_scatter_rows) {
    size_type const output_row = scatter_map[row];

    if (mark_true) {
      destination.set_valid(output_row);
    } else {
      destination.set_null(output_row);
    }

    row += stride;
  }
}

template <typename MapIterator>
void scatter_scalar_bitmask_inplace(std::reference_wrapper<scalar const> const& source,
                                    MapIterator scatter_map,
                                    size_type num_scatter_rows,
                                    column& target,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  constexpr size_type block_size = 256;
  size_type const grid_size      = grid_1d(num_scatter_rows, block_size).num_blocks;

  auto const source_is_valid = source.get().is_valid(stream);
  if (target.nullable() or not source_is_valid) {
    if (not target.nullable()) {
      // Target must have a null mask if the source is not valid
      auto mask = detail::create_null_mask(target.size(), mask_state::ALL_VALID, stream, mr);
      target.set_null_mask(std::move(mask), 0);
    }

    auto target_view = mutable_column_device_view::create(target, stream);

    auto bitmask_kernel = source_is_valid ? marking_bitmask_kernel<true, decltype(scatter_map)>
                                          : marking_bitmask_kernel<false, decltype(scatter_map)>;
    bitmask_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      *target_view, scatter_map, num_scatter_rows);

    target.set_null_count(
      cudf::detail::null_count(target.view().null_mask(), 0, target.size(), stream));
  }
}

template <typename Element, typename MapIterator>
struct column_scalar_scatterer_impl {
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(source.get().type() == target.type(),
                 "scalar and column types must match",
                 cudf::data_type_error);

    // make a copy of data and null mask from source
    auto result      = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // Use permutation iterator with constant index to dereference scalar data
    auto scalar_impl = static_cast<scalar_type_t<Element> const*>(&source.get());
    auto scalar_iter =
      thrust::make_permutation_iterator(scalar_impl->data(), thrust::make_constant_iterator(0));

    thrust::scatter(rmm::exec_policy_nosync(stream),
                    scalar_iter,
                    scalar_iter + scatter_rows,
                    scatter_iter,
                    result_view.begin<Element>());

    scatter_scalar_bitmask_inplace(source, scatter_iter, scatter_rows, *result, stream, mr);
    return result;
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<string_view, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(source.get().type() == target.type(), "scalar and column types must match");

    auto const scalar_impl = static_cast<string_scalar const*>(&source.get());
    auto const source_view = string_view(scalar_impl->data(), scalar_impl->size());
    auto const begin       = thrust::make_constant_iterator(source_view);
    auto const end         = begin + scatter_rows;
    auto result            = strings::detail::scatter(begin, end, scatter_iter, target, stream, mr);

    scatter_scalar_bitmask_inplace(source, scatter_iter, scatter_rows, *result, stream, mr);
    return result;
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<list_view, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result =
      lists::detail::scatter(source, scatter_iter, scatter_iter + scatter_rows, target, stream, mr);

    scatter_scalar_bitmask_inplace(source, scatter_iter, scatter_rows, *result, stream, mr);
    return result;
  }
};

template <typename MapIterator>
struct column_scalar_scatterer_impl<dictionary32, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
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
    auto scalar_index = dictionary::detail::get_index(
      dict_view, source.get(), stream, rmm::mr::get_current_device_resource());
    auto scalar_iter = thrust::make_permutation_iterator(
      indexalator_factory::make_input_iterator(*scalar_index), thrust::make_constant_iterator(0));
    auto new_indices = std::make_unique<column>(dict_view.get_indices_annotated(), stream, mr);
    auto target_iter = indexalator_factory::make_output_iterator(new_indices->mutable_view());

    thrust::scatter(rmm::exec_policy_nosync(stream),
                    scalar_iter,
                    scalar_iter + scatter_rows,
                    scatter_iter,
                    target_iter);

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
    auto result = make_dictionary_column(std::move(keys_column),
                                         std::move(indices_column),
                                         std::move(*(contents.null_mask.release())),
                                         null_count);

    scatter_scalar_bitmask_inplace(source, scatter_iter, scatter_rows, *result, stream, mr);
    return result;
  }
};

template <typename MapIterator>
struct column_scalar_scatterer {
  template <typename Element>
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
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

template <typename MapIterator>
struct column_scalar_scatterer_impl<struct_view, MapIterator> {
  std::unique_ptr<column> operator()(std::reference_wrapper<scalar const> const& source,
                                     MapIterator scatter_iter,
                                     size_type scatter_rows,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    // For each field of `source`, copy construct a scalar from the field
    // and dispatch to the corresponding scalar scatterer

    auto typed_s             = static_cast<struct_scalar const*>(&source.get());
    size_type const n_fields = typed_s->view().num_columns();
    CUDF_EXPECTS(n_fields == target.num_children(), "Mismatched number of fields.");

    auto scatter_functor   = column_scalar_scatterer<decltype(scatter_iter)>{};
    auto fields_iter_begin = make_counting_transform_iterator(0, [&](auto const& i) {
      auto row_slr = detail::get_element(
        typed_s->view().column(i), 0, stream, rmm::mr::get_current_device_resource());
      return type_dispatcher<dispatch_storage_type>(row_slr->type(),
                                                    scatter_functor,
                                                    *row_slr,
                                                    scatter_iter,
                                                    scatter_rows,
                                                    target.child(i),
                                                    stream,
                                                    mr);
    });
    std::vector<std::unique_ptr<column>> fields(fields_iter_begin, fields_iter_begin + n_fields);

    // Compute null mask
    rmm::device_buffer null_mask =
      target.nullable()
        ? detail::copy_bitmask(target, stream, mr)
        : detail::create_null_mask(target.size(), mask_state::UNALLOCATED, stream, mr);
    column null_mask_stub(data_type{type_id::STRUCT},
                          target.size(),
                          rmm::device_buffer{},
                          std::move(null_mask),
                          target.null_count());
    scatter_scalar_bitmask_inplace(source, scatter_iter, scatter_rows, null_mask_stub, stream, mr);
    size_type null_count = null_mask_stub.null_count();
    auto contents        = null_mask_stub.release();

    // Null mask pushdown inside factory method
    return make_structs_column(
      target.size(), std::move(fields), null_count, std::move(*contents.null_mask), stream, mr);
  }
};

}  // namespace

std::unique_ptr<table> scatter(table_view const& source,
                               column_view const& scatter_map,
                               table_view const& target,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(source.num_columns() == target.num_columns(),
               "Number of columns in source and target not equal",
               std::invalid_argument);
  CUDF_EXPECTS(scatter_map.size() <= source.num_rows(),
               "Size of scatter map must be equal to or less than source rows",
               std::invalid_argument);
  CUDF_EXPECTS(std::equal(source.begin(),
                          source.end(),
                          target.begin(),
                          [](auto const& col1, auto const& col2) {
                            return col1.type().id() == col2.type().id();
                          }),
               "Column types do not match between source and target",
               cudf::data_type_error);
  CUDF_EXPECTS(not scatter_map.has_nulls(), "Scatter map contains nulls", std::invalid_argument);

  if (scatter_map.is_empty()) { return std::make_unique<table>(target, stream, mr); }

  // create index type normalizing iterator for the scatter_map
  auto map_begin = indexalator_factory::make_input_iterator(scatter_map);
  auto map_end   = map_begin + scatter_map.size();
  return detail::scatter(source, map_begin, map_end, target, stream, mr);
}

std::unique_ptr<table> scatter(table_view const& source,
                               device_span<size_type const> const scatter_map,
                               table_view const& target,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(scatter_map.size() <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "scatter map size exceeds the column size limit",
               std::overflow_error);
  auto map_col = column_view(data_type{type_to_id<size_type>()},
                             static_cast<size_type>(scatter_map.size()),
                             scatter_map.data(),
                             nullptr,
                             0);
  return detail::scatter(source, map_col, target, stream, mr);
}

std::unique_ptr<table> scatter(std::vector<std::reference_wrapper<scalar const>> const& source,
                               column_view const& indices,
                               table_view const& target,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(source.size() == static_cast<size_t>(target.num_columns()),
               "Number of scalars in source and number of columns in target not equal",
               std::invalid_argument);
  CUDF_EXPECTS(not indices.has_nulls(), "indices contains nulls", std::invalid_argument);

  if (indices.is_empty()) { return std::make_unique<table>(target, stream, mr); }

  // Create normalizing iterator for indices column
  auto map_begin = indexalator_factory::make_input_iterator(indices);

  // Optionally check map index values are within the number of target rows.
  auto const n_rows = target.num_rows();

  // Transform negative indices to index + target size
  auto scatter_rows = indices.size();
  // note: the intermediate ((in % n_rows) + n_rows) will overflow a size_type for any value of `in`
  // > (2^31)/2, but the end result after the final (% n_rows) will fit. so we'll do the computation
  // using a signed 64 bit value.
  auto scatter_iter = thrust::make_transform_iterator(
    map_begin,
    cuda::proclaim_return_type<size_type>(
      [n_rows = static_cast<int64_t>(n_rows)] __device__(size_type in) -> size_type {
        return static_cast<size_type>(((static_cast<int64_t>(in) % n_rows) + n_rows) % n_rows);
      }));

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

  thrust::sequence(rmm::exec_policy_nosync(stream),
                   mutable_indices.begin<size_type>(),
                   mutable_indices.end<size_type>(),
                   0);

  // The scatter map is actually a table with only one column, which is scatter map.
  auto scatter_map = detail::apply_boolean_mask(
    table_view{{indices->view()}}, boolean_mask, stream, rmm::mr::get_current_device_resource());
  auto output_table = detail::scatter(
    table_view{{input}}, scatter_map->get_column(0).view(), table_view{{target}}, stream, mr);

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
               "Mismatch in number of input columns and target columns",
               std::invalid_argument);
  CUDF_EXPECTS(boolean_mask.size() == target.num_rows(),
               "Boolean mask size and number of target rows mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8,
               "Mask must be of Boolean type",
               cudf::data_type_error);
  // Count valid pair of input and columns as per type at each column index i
  CUDF_EXPECTS(
    std::all_of(thrust::counting_iterator<size_type>(0),
                thrust::counting_iterator<size_type>(target.num_columns()),
                [&input, &target](auto index) {
                  return ((input.column(index).type().id()) == (target.column(index).type().id()));
                }),
    "Type mismatch in input column and target column",
    cudf::data_type_error);

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
  std::vector<std::reference_wrapper<scalar const>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(static_cast<size_type>(input.size()) == target.num_columns(),
               "Mismatch in number of scalars and target columns",
               std::invalid_argument);
  CUDF_EXPECTS(boolean_mask.size() == target.num_rows(),
               "Boolean mask size and number of target rows mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8,
               "Mask must be of Boolean type",
               cudf::data_type_error);

  // Count valid pair of input and columns as per type at each column/scalar index i
  CUDF_EXPECTS(
    std::all_of(thrust::counting_iterator<size_type>(0),
                thrust::counting_iterator<size_type>(target.num_columns()),
                [&input, &target](auto index) {
                  return (input[index].get().type().id() == target.column(index).type().id());
                }),
    "Type mismatch in input scalar and target column",
    cudf::data_type_error);

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
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scatter(source, scatter_map, target, stream, mr);
}

std::unique_ptr<table> scatter(std::vector<std::reference_wrapper<scalar const>> const& source,
                               column_view const& indices,
                               table_view const& target,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scatter(source, indices, target, stream, mr);
}

std::unique_ptr<table> boolean_mask_scatter(table_view const& input,
                                            table_view const& target,
                                            column_view const& boolean_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::boolean_mask_scatter(input, target, boolean_mask, stream, mr);
}

std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<scalar const>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::boolean_mask_scatter(input, target, boolean_mask, stream, mr);
}

}  // namespace cudf
