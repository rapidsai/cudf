/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/search.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <cudf/table/experimental/row_operators.cuh>

#include <hash/unordered_multiset.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

namespace cudf {
namespace {
std::unique_ptr<column> search_bound(table_view const& t,
                                     table_view const& values,
                                     bool find_lower_bound,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    column_order.empty() or static_cast<std::size_t>(t.num_columns()) == column_order.size(),
    "Mismatch between number of columns and column order.");
  CUDF_EXPECTS(
    null_precedence.empty() or static_cast<std::size_t>(t.num_columns()) == null_precedence.size(),
    "Mismatch between number of columns and null precedence.");

  // Allocate result column
  auto result = make_numeric_column(
    data_type{type_to_id<size_type>()}, values.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto const out_it = result->mutable_view().data<size_type>();

  // Handle empty inputs
  if (t.num_rows() == 0) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(out_it, 0, values.num_rows() * sizeof(size_type), stream.value()));
    return result;
  }

  // This utility will ensure all corresponding dictionary columns have matching keys.
  // It will return any new dictionary columns created as well as updated table_views.
  auto const matched = dictionary::detail::match_dictionaries({t, values}, stream);

  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  auto const& lhs     = find_lower_bound ? matched.second.front() : matched.second.back();
  auto const& rhs     = find_lower_bound ? matched.second.back() : matched.second.front();
  auto const comp     = cudf::experimental::row::lexicographic::table_comparator(
    lhs, rhs, column_order, null_precedence, stream);
  auto const has_any_nulls = has_nested_nulls(t) or has_nested_nulls(values);
  auto const dcomp         = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

  auto const do_search = [find_lower_bound](auto&&... args) {
    if (find_lower_bound) {
      thrust::lower_bound(std::forward<decltype(args)>(args)...);
    } else {
      thrust::upper_bound(std::forward<decltype(args)>(args)...);
    }
  };

  do_search(rmm::exec_policy(stream),
            count_it,
            count_it + t.num_rows(),
            count_it,
            count_it + values.num_rows(),
            out_it,
            dcomp);

  return result;
}

struct contains_scalar_dispatch {
  template <typename Element>
  bool operator()(column_view const& col, scalar const& value, rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(col.type() == value.type(), "scalar and column types must match");

    using Type       = device_storage_type_t<Element>;
    using ScalarType = cudf::scalar_type_t<Element>;
    auto d_col       = column_device_view::create(col, stream);
    auto s           = static_cast<const ScalarType*>(&value);

    if (col.has_nulls()) {
      auto found_iter = thrust::find(rmm::exec_policy(stream),
                                     d_col->pair_begin<Type, true>(),
                                     d_col->pair_end<Type, true>(),
                                     thrust::make_pair(s->value(stream), true));

      return found_iter != d_col->pair_end<Type, true>();
    } else {
      auto found_iter = thrust::find(rmm::exec_policy(stream),  //
                                     d_col->begin<Type>(),
                                     d_col->end<Type>(),
                                     s->value(stream));

      return found_iter != d_col->end<Type>();
    }
  }
};

template <>
bool contains_scalar_dispatch::operator()<cudf::list_view>(column_view const&,
                                                           scalar const&,
                                                           rmm::cuda_stream_view)
{
  CUDF_FAIL("list_view type not supported yet");
}

template <>
bool contains_scalar_dispatch::operator()<cudf::struct_view>(column_view const& col,
                                                             scalar const& value,
                                                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(col.type() == value.type(), "scalar and column types must match");

  auto const scalar_tview = static_cast<struct_scalar const*>(&value)->view();
  CUDF_EXPECTS(col.num_children() == scalar_tview.num_columns(),
               "struct scalar and structs column must have the same number of children");
  for (size_type i = 0; i < col.num_children(); ++i) {
    CUDF_EXPECTS(col.child(i).type() == scalar_tview.column(i).type(),
                 "scalar and column children types must match");
  }

  // Convert the input scalar value into a structs column of one row.
  auto const val_col = cudf::make_structs_column(
    1, table{scalar_tview}.release(), 0, rmm::device_buffer{0, stream}, stream);

  auto const col_tview     = table_view{{col}};
  auto const has_any_nulls = has_nested_nulls(col_tview) || has_nested_nulls(scalar_tview);

  auto const comp = cudf::experimental::row::equality::table_comparator(
    col_tview, table_view{{val_col->view()}}, stream);
  auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

#if 0
  // Prepare to flatten the structs column and scalar.
  auto const flatten_nullability = has_any_nulls
                                     ? structs::detail::column_nullability::FORCE
                                     : structs::detail::column_nullability::MATCH_INCOMING;

  // Flatten the input structs column, only materialize the bitmask if there is null in the input.
  auto const col_flattened =
    structs::detail::flatten_nested_columns(col_tview, {}, {}, flatten_nullability);
  auto const val_flattened =
    structs::detail::flatten_nested_columns(scalar_tview, {}, {}, flatten_nullability);

  // The struct scalar only contains the struct member columns.
  // Thus, if there is any null in the input, we must exclude the first column in the flattened
  // table of the input column from searching because that column is the materialized bitmask of
  // the input structs column.
  auto const col_flattened_content  = col_flattened.flattened_columns();
  auto const col_flattened_children = table_view{
    std::vector<column_view>{col_flattened_content.begin() + static_cast<size_type>(has_any_nulls),
                             col_flattened_content.end()}};

  auto const d_col_children_ptr = table_device_view::create(col_flattened_children, stream);
  auto const d_val_ptr          = table_device_view::create(val_flattened, stream);

  auto const comp       = row_equality_comparator(
    nullate::DYNAMIC{has_any_nulls}, *d_col_children_ptr, *d_val_ptr, null_equality::EQUAL);
#endif
  auto const d_col_ptr  = column_device_view::create(col, stream);
  auto const start_iter = thrust::make_counting_iterator<size_type>(0);
  auto const end_iter   = start_iter + col.size();
  auto const found_iter = thrust::find_if(
    rmm::exec_policy(stream),
    start_iter,
    end_iter,
    [dcomp, d_col = *d_col_ptr, has_null_structs = col.has_nulls()] __device__(auto const idx) {
      if (has_null_structs && d_col.is_null_nocheck(idx)) { return false; }
      return dcomp(idx, 0);  // compare col[idx] == val[0].
    });

  return found_iter != end_iter;
}

template <>
bool contains_scalar_dispatch::operator()<cudf::dictionary32>(column_view const& col,
                                                              scalar const& value,
                                                              rmm::cuda_stream_view stream)
{
  auto dict_col = cudf::dictionary_column_view(col);
  // first, find the value in the dictionary's key set
  auto index = cudf::dictionary::detail::get_index(dict_col, value, stream);
  // if found, check the index is actually in the indices column
  return index->is_valid(stream) ? cudf::type_dispatcher(dict_col.indices().type(),
                                                         contains_scalar_dispatch{},
                                                         dict_col.indices(),
                                                         *index,
                                                         stream)
                                 : false;
}

}  // namespace

namespace detail {
bool contains(column_view const& col, scalar const& value, rmm::cuda_stream_view stream)
{
  if (col.is_empty()) { return false; }
  if (not value.is_valid(stream)) { return col.has_nulls(); }

  return cudf::type_dispatcher(col.type(), contains_scalar_dispatch{}, col, value, stream);
}

struct multi_contains_dispatch {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    std::unique_ptr<column> result = make_numeric_column(data_type{type_to_id<bool>()},
                                                         haystack.size(),
                                                         copy_bitmask(haystack),
                                                         haystack.null_count(),
                                                         stream,
                                                         mr);

    if (haystack.is_empty()) { return result; }

    mutable_column_view result_view = result.get()->mutable_view();

    if (needles.is_empty()) {
      thrust::fill(
        rmm::exec_policy(stream), result_view.begin<bool>(), result_view.end<bool>(), false);
      return result;
    }

    auto hash_set        = cudf::detail::unordered_multiset<Element>::create(needles, stream);
    auto device_hash_set = hash_set.to_device();

    auto d_haystack_ptr = column_device_view::create(haystack, stream);
    auto d_haystack     = *d_haystack_ptr;

    if (haystack.has_nulls()) {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_haystack] __device__(size_t index) {
                          return d_haystack.is_null_nocheck(index) ||
                                 device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    } else {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_haystack] __device__(size_t index) {
                          return device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    }

    return result;
  }
};

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<list_view>(
  column_view const&, column_view const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<struct_view>(
  column_view const&, column_view const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
{
  CUDF_FAIL("struct_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto haystack_matched    = dictionary::detail::add_keys(haystack, needles.keys(), stream);
  auto const haystack_view = dictionary_column_view(haystack_matched->view());
  auto needles_matched     = dictionary::detail::set_keys(needles, haystack_view.keys(), stream);
  auto const needles_view  = dictionary_column_view(needles_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  return cudf::type_dispatcher(haystack_indices.type(),
                               multi_contains_dispatch{},
                               haystack_indices,
                               needles_indices,
                               stream,
                               mr);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::type_dispatcher(
    haystack.type(), multi_contains_dispatch{}, haystack, needles, stream, mr);
}

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return search_bound(t, values, true, column_order, null_precedence, stream, mr);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return search_bound(t, values, false, column_order, null_precedence, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::lower_bound(
    t, values, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::upper_bound(
    t, values, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

bool contains(column_view const& col, scalar const& value)
{
  CUDF_FUNC_RANGE();
  return detail::contains(col, value, rmm::cuda_stream_default);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
